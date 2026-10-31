Improve how tiles reach shared memory. Pipeline the main loop's loads with `num_stages` of 2 to 4
so cp.async overlaps loads with compute, within the shared-memory budget per block. Use TMA
tensor descriptors only if the device supports them and a measurement shows they help; on sm_120
they measured at parity with pointer loads.
