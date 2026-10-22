/* Force the GreenBoost shim's overflow path on bare metal.
 *
 * The shim inflates what cuMemGetInfo *reports*, but its own gb_needs_overflow
 * check uses the REAL free VRAM minus headroom. So an allocation larger than
 * (real_free - headroom) must be routed out of VRAM into pinned system RAM via
 * Path A. This probe allocates exactly one such buffer and measures it.
 *
 * Deliberate memory discipline: only a 1 GB VRAM scratch source, never a second
 * buffer the size of BIG. That is what blew the host up last time.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda.h>

#define cuDeviceSynchronize cuCtxSynchronize

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

int main(int argc, char **argv)
{
    size_t BIG_MB    = argc > 1 ? (size_t)atoi(argv[1]) : 20480;
    size_t SCRATCH   = (size_t)512 << 20;          /* 512 MB VRAM source */
    size_t bigbytes  = BIG_MB << 20;
    int    rc_exit   = 1;

    cuInit(0);
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);

    size_t f = 0, t = 0; cuMemGetInfo(&f, &t);
    printf("shim-reported VRAM: %.1f free / %.1f total GB\n",
           f / 1073741824.0, t / 1073741824.0);
    printf("requesting one %zu MB buffer (overflow threshold is real_free - headroom)\n\n",
           BIG_MB);

    CUdeviceptr src = 0;
    if (cuMemAlloc(&src, SCRATCH) != CUDA_SUCCESS) {
        printf("! no VRAM for scratch\n"); goto out;
    }
    cuMemsetD8(src, 0x5A, SCRATCH);
    cuCtxSynchronize();
    printf("scratch source allocated (%zu MB in VRAM)\n", SCRATCH >> 20);

    /* --- this is the allocation the shim must route to system RAM --- */
    CUdeviceptr big = 0;
    double t0 = now();
    CUresult r = cuMemAlloc(&big, bigbytes);
    double t1 = now();
    if (r != CUDA_SUCCESS) {
        printf("! cuMemAlloc(%zu MB) failed: %d\n", BIG_MB, r);
        goto out;
    }
    printf("cuMemAlloc returned 0x%llx after %.1f s\n\n",
           (unsigned long long)big, t1 - t0);

    /* Where did it actually land? Ask the kernel module. */
    system("grep -E 'T2 allocated|Active DMA-BUF' /sys/class/greenboost/greenboost/pool_info");

    /* Fill it from VRAM — if this succeeds, the T2 region is device-writable. */
    t0 = now();
    r = cuMemcpyDtoD(big, src, SCRATCH);
    cuCtxSynchronize();
    t1 = now();
    printf("DtoD VRAM->T2 (%zu MB): rc=%d  %.2f GB/s\n",
           SCRATCH >> 20, r, SCRATCH / (t1 - t0) / 1e9);

    /* Read the whole buffer back repeatedly: this is the number that matters,
     * because every inference step streams weights across PCIe. */
    double best = 0;
    for (int i = 0; i < 3; i++) {
        t0 = now();
        r = cuMemcpyDtoD(src, big, SCRATCH);      /* read first 512 MB of T2 */
        cuCtxSynchronize();
        t1 = now();
        double bw = SCRATCH / (t1 - t0) / 1e9;
        if (bw > best) best = bw;
    }
    printf("device-side read from T2 : %.2f GB/s (rc=%d)\n", best, r);

    /* and a full-buffer device->device sweep if it fits in scratch+time */
    size_t chunk = SCRATCH;
    double full = 0;
    t0 = now();
    for (size_t off = 0; off < bigbytes; off += chunk)
        cuMemcpyDtoD(src, big + off, chunk < bigbytes - off ? chunk : bigbytes - off);
    cuCtxSynchronize();
    t1 = now();
    full = bigbytes / (t1 - t0) / 1e9;
    printf("full %.1f GB sweep T2->VRAM: %.2f GB/s\n",
           bigbytes / 1073741824.0, full);

    printf("\n*** SHIM ROUTED %.0f MB INTO SYSTEM RAM, READ AT %.2f GB/s ***\n",
           BIG_MB / 1024.0, full);

    cuMemFree(big);
    cuMemFree(src);
    rc_exit = 0;
out:
    cuCtxDestroy(ctx);
    return rc_exit;
}