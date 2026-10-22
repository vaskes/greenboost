/* Bare-metal probe for GreenBoost — Path A via cuMemHostRegister.
 *
 * v2 fixes the nm() static-buffer aliasing bug from v1 (two calls in one printf
 * both printed the first result, so "GetDevPtr" status was never actually seen).
 * Path A is now the main event: the whole point is real PCIe bandwidth into
 * pinned system RAM, not just "did it return SUCCESS".
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <time.h>
#include <cuda.h>
#include "greenboost_ioctl.h"

#define cuDeviceSynchronize cuCtxSynchronize

/* Ring of buffers so several nm() results can appear in one printf. */
static char  nbuf[8][96];
static int   nslot;
static const char *nm(CUresult r, const char *what)
{
    const char *n = "?";
    char *b = nbuf[nslot++ & 7];
    cuGetErrorName(r, &n);
    snprintf(b, 96, "%s=%d(%s)", what, r, n);
    return b;
}

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

int main(int argc, char **argv)
{
    size_t MB = argc > 1 ? (size_t)atoi(argv[1]) : 512;
    size_t bytes = MB << 20;
    int iters = argc > 2 ? atoi(argv[2]) : 3;
    int rc_exit = 1;

    if (cuInit(0) != CUDA_SUCCESS) { printf("cuInit failed\n"); return 1; }
    CUdevice dev;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) { printf("cuDeviceGet failed\n"); return 1; }
    CUcontext ctx;
    if (cuCtxCreate(&ctx, 0, dev) != CUDA_SUCCESS) { printf("cuCtxCreate failed\n"); return 1; }

    int drv = 0; cuDriverGetVersion(&drv);
    size_t dfree = 0, dtotal = 0; cuMemGetInfo(&dfree, &dtotal);
    printf("CUDA driver %d | dev %d | VRAM %.1f/%.1f GB free | test %zu MB x%d\n\n",
           drv, dev, dfree / 1073741824.0, dtotal / 1073741824.0, MB, iters);

    /* ---------- A: HostRegister + GetDevicePointer, status shown properly ---------- */
    printf("=== A. cuMemHostRegister -> cuMemHostGetDevicePointer ===\n");
    int cold = getenv("GB_COLD") != NULL;   /* mimic the shim: untouched mmap */
    int usepin = getenv("GB_PIN") != NULL;  /* mimic the shim: pin via greenboost.ko */
    void *hm = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | (cold ? 0 : MAP_POPULATE), -1, 0);
    if (hm == MAP_FAILED) { perror("mmap"); goto out; }
    if (!cold) memset(hm, 0xAB, bytes);

    int pinfd = -1;
    if (usepin) {
        /* Exactly what the shim does: mmap anonymous, then ask greenboost.ko
         * to pin those user pages and export them as a DMA-BUF. */
        pinfd = open("/dev/greenboost", O_RDWR);
        if (pinfd < 0) { perror("open /dev/greenboost"); goto out; }
        struct gb_pin_req pr;
        memset(&pr, 0, sizeof pr);
        pr.vaddr = (uint64_t)(uintptr_t)hm;
        pr.size  = bytes;
        pr.flags = GB_ALLOC_WEIGHTS;
        pr.fd    = -1;
        if (ioctl(pinfd, GB_IOCTL_PIN_USER_PTR, &pr) < 0) {
            perror("GB_IOCTL_PIN_USER_PTR");
            close(pinfd); goto out;
        }
        printf("   GB_IOCTL_PIN_USER_PTR ok -> dmabuf fd=%d (%zu MB pinned)\n",
               pr.fd, pr.size >> 20);
    }
    printf("   host mmap %p (%s%s)\n", hm,
           cold ? "COLD" : "warm",
           usepin ? " + PINNED BY greenboost.ko" : "");

    static const unsigned flagset[] = { 0x02, 0x00, 0x01 };
    CUresult rreg = CUDA_ERROR_UNKNOWN, rptr = CUDA_ERROR_UNKNOWN;
    CUdeviceptr devptr = 0;
    for (int f = 0; f < 3; f++) {
        CUdeviceptr dp = 0;
        CUresult r1 = cuMemHostRegister(hm, bytes, flagset[f]);
        CUresult r2 = (r1 == CUDA_SUCCESS) ? cuMemHostGetDevicePointer(&dp, hm, 0)
                                           : CUDA_ERROR_INVALID_VALUE;
        printf("   flags=0x%02x  %s  %s  devptr=0x%llx %s\n",
               flagset[f], nm(r1, "Register"), nm(r2, "GetDevPtr"),
               (unsigned long long)dp, dp == (CUdeviceptr)hm ? "== hostptr (UVA)" : "");
        if (r1 == CUDA_SUCCESS) cuMemHostUnregister(hm);
    }

    /* Register for real, keep it, measure. DEVICEMAP is what the shim uses. */
    rreg = cuMemHostRegister(hm, bytes, 0x02);
    rptr = cuMemHostGetDevicePointer(&devptr, hm, 0);
    printf("   final: %s  %s  devptr=0x%llx\n", nm(rreg, "Register"),
           nm(rptr, "GetDevPtr"), (unsigned long long)devptr);
    if (rptr != CUDA_SUCCESS) {
        printf("\n*** CONFIRMED: %s breaks GetDevPtr ***\n",
               usepin ? "kernel-pinned mapping" : "cold mapping");
        rc_exit = 7; goto out;
    }
    if (pinfd >= 0) { close(pinfd); pinfd = -1; }
    if (cold) memset(hm, 0xAB, bytes);   /* touch it only AFTER registering */

    if (getenv("GB_NOBW")) {
        printf("\n(GB_NOBW: skipping bandwidth, VRAM scratch would not fit)\n");
        rc_exit = 0; goto out;
    }
    /* ---------- bandwidth through the registered region ---------- */
    CUdeviceptr src = 0;
    if (cuMemAlloc(&src, bytes) != CUDA_SUCCESS) {
        printf("   ! no free VRAM for source buffer\n"); goto out;
    }
    cuMemsetD8(src, 0x5A, bytes);
    cuCtxSynchronize();

    double bh2d = 0, bd2h = 0, bcopy = 0;
    CUdeviceptr dst = 0;
    int have_dst = cuMemAlloc(&dst, bytes) == CUDA_SUCCESS;

    for (int i = 0; i < iters; i++) {
        double t0 = now();
        cuMemcpyHtoD(devptr, (const void *)src, bytes);          /* VRAM -> pinned system RAM */
        cuCtxSynchronize();
        double t1 = now();
        if (t1 - t0 < bh2d || bh2d == 0) bh2d = t1 - t0;

        t0 = now();
        cuMemcpyDtoH((void *)hm, devptr, bytes);   /* pinned system RAM -> host */
        cuCtxSynchronize();
        t1 = now();
        if (t1 - t0 < bd2h || bd2h == 0) bd2h = t1 - t0;

        if (have_dst) {
            t0 = now();
            cuMemcpyDtoD(dst, devptr, bytes);      /* read T2 on device side */
            cuCtxSynchronize();
            t1 = now();
            if (t1 - t0 < bcopy || bcopy == 0) bcopy = t1 - t0;
        }
    }

    printf("\n=== B. bandwidth through Path A (pinned system RAM) ===\n");
    printf("   VRAM -> system RAM (H2D)      %7.2f GB/s\n", bytes / bh2d / 1e9);
    printf("   system RAM -> host  (D2H)      %7.2f GB/s\n", bytes / bd2h / 1e9);
    if (have_dst)
        printf("   system RAM read on device     %7.2f GB/s\n", bytes / bcopy / 1e9);
    printf("   (PCIe 3.0 x16 theoretical ~12.5 GB/s)\n");

    cuMemFree(src);
    if (have_dst) cuMemFree(dst);

    /* ---------- C: what the shim's fd path does, for the record ---------- */
    printf("\n=== C. DMA-BUF fd -> cuImportExternalMemory (shim's unused path) ===\n");
    int gbfd = open("/dev/greenboost", O_RDWR);
    if (gbfd >= 0) {
        struct gb_alloc_req ar;
        memset(&ar, 0, sizeof ar);
        ar.size = bytes; ar.fd = -1; ar.flags = GB_ALLOC_WEIGHTS;
        if (ioctl(gbfd, GB_IOCTL_ALLOC, &ar) == 0) {
            printf("   GB_IOCTL_ALLOC -> dmabuf fd=%d\n", ar.fd);
            CUDA_EXTERNAL_MEMORY_HANDLE_DESC hd;
            memset(&hd, 0, sizeof hd);
            hd.type = 1; hd.handle.fd = ar.fd; hd.size = bytes; hd.flags = 0;
            CUexternalMemory ext = (CUexternalMemory)0;
            CUresult r = cuImportExternalMemory(&ext, &hd);
            printf("   %s\n", nm(r, "ImportExternalMemory"));
            if (r == CUDA_SUCCESS) cuDestroyExternalMemory(ext);
        } else perror("   GB_IOCTL_ALLOC");
        close(gbfd);
    }

    printf("\n*** PATH A USABLE ON BARE METAL ***\n");
    rc_exit = 0;
out:
    cuCtxDestroy(ctx);
    return rc_exit;
}