# Sting

An updated well written python package for analysis of mother-machine images with test coverage, neural network training and documentation.

A package that will be developed to a production quality.


## Massive Upgrade. TODO list (before 25.12.2023)

- [ ] Upgrade cell segmentation to achieve real-time omnipose/ [affinity segmentation](https://github.com/ryanirl/torchvf) on 4096 x 1024 images in < 100 ms on RTX 4090
- [ ] Prune and do model level optimizations to achieve this latency limits
- [ ] Containerize the segmentation models, Barcode detection models using nvidia triton inference server.
- [ ] Communication layer between Microscope PC and long-term storage + additional compute with MATLAB on the cluster.
- [ ] Figure out faster clustering in the last steps of the instance segmentation model.
- [ ] Registration of barcodes to align channels more robustly and some scheme to keep track of drifts.
- [ ] Bright field support (maybe).
- [ ] Sophisticated event generation and preview (Easy to do).
- [ ] See if you can compile baxter type dynamic programming based tracking or a [bayesian tracker](https://github.com/quantumjot/btrack)
- [ ] Backbone generation right after segmentation and effects on cell size. See if a network output or post-processed output can be useful here.
- [ ] Dot detection intergration on the cluster (most likely) or some simple GPU implementation of wavelet methods if doable
- [ ] Fork plot construction on the cluster and request results over the network.
- [ ] More UI options to support Fork plotting and other data fetching from the cluster. 