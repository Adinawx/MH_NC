# Blank Space: Adaptive Causal Network Coding for Multi-Hop Networks

This repository contains the implementation of the **Blank Space AC-RLNC** algorithm, a novel Adaptive and Causal Network Coding solution designed to mitigate the triplet trade-off between throughput-delay-efficiency in multi-hop networks. The full algorithm is described in our paper: [Blank Space: Adaptive Causal Coding for Streaming Communications Over Multi-Hop Networks](https://arxiv.org/abs/2502.11984)

### Project Structure

```
├── Project/                  # Project Folder

  ├── Data/                   # Data Folder
    ├── BEC/                  # Binary Erasure Channel patterns
    │   └── ch_n/             # Channel n patterns
    ├── GE_p_X_g_Y_b_Z/       # Gilbert-Elliot models
    │   └── ch_n/             # Channel n patterns
    └── ber_series_less/      # Bernoulli input distributions
        └── rate_X/           # Rate X patterns

  ├── Code/                      # Code Folder
    ├── main.py                  # Main file - calls run\run_all.py\run_main()
    ├── utils/                   # Configuration setup
    │   ├── config.py            # Input parameters in "param"
    │   └── config_setup.py      # Conversion from JSON
    ├── run/                     # Main running functions
    │   ├── run_1.py             # Single simulation run
    │   ├── run_all.py           # Monte Carlo simulation (rep times) - This is the main file
    │   ├── data_storage.py      # Data storage class
    │   └── plot_results.py      # Results visualization
    ├── ns/                      # Network system implementation
    │   └── nc_enc.py            # Connection between code and encoder
    │   └── ...                  # More Code
    ├── erasures_generator/      # Channel model generation
    │   ├── erasure_generate.py  # Channel erasure pattern generation
    │   └── ber_input_series.py  # Bernoulli input generation
    └── acrlnc_node/             # AC-RLNC implementation
        ├── encoder.py           # Network coding encoder
        ├── ac_node.py           # Node implementation
        ├── encoder_mix_all.py   # Simplified encoder for MIXALL variant
        └── ac_node_mix_all.py   # MIXALL variant implementation

  ├── Results/                    # Results Folder, If does not exists, created during run
    ├── run_123/                  # Base run name
    │   ├── RTT=20_ER_I=0/        # RTT and varying erasure rate settings
    │   │   ├── ff_log.txt        # Detailed transmission logs
    │   │   ├── metrics.txt       # Overall performance metrics
    │   │   └── metrics_nodes.txt # Per-node metrics
    │   └── ...
    └── ...
```

*Further details on the imlementation can be found in the .docx file.*

## Running the Code

### Initial Setup

1. Generate Data for channels using `erasure_generator/erasure_generate.py`
2. Generate Bernoulli inputs in `erasure_generator/ber_input_series.py`
3. Set parameters in `config/config.py`
   - Project Folder Path
   - Results Folder Path

### Run Modes

1. Set simulation type in `run/run_all.py/run_main()` function -- this is the "main"
2. Set parameters in `config/config.py`

> **Note:**
> 
> 1. For each simulation you must choose another run name:  
>    `"results_filename_base": "run1", # results filename - must change each run`
> 
> 2. Plot_results for paper -- choose the `"results_filename_base"`:
>    - Must include all three: MIXALL, BS-EMPTY, AC-FEC
>    - In this version must have SR-ARQ results (not included in GitHub)

### Algorithm Variants
Configure in `run_all.py/main_run()`
1. Baseline: `MIXALL` -
    - calls the `*_mix_all` functions in acrlnc folder.
    - Implement AC-RLNC at the source only (H_0)
    - Any intermediate node “mix all” the packets in the buffer and forwards 
2. All other options inplement NET-AC-RLNC with different variants.
    - calls the not mix_all functions in acrlnc folder.

| Variant | Description |
|---------|-------------|
| `AC-FEC` | Both pause mechanisma are disabled (FEC instead of pause (Empty) in No-New No-FEC options). |
| `AC-EMPTY` |No-New No-FEC enabled, Blank Space disabled |
| `BS-FEC` | No-New No-FEC disabled, Blank Space enabled |
| `BS-EMPTY` | Both pause mechanisms enabled (full algorithm) |



## Implementation Details

### Data

1. **Information Data**
   - No actual data transmitted
   - Packets represented by indices only

2. **Erasure Channels**
   - Represented by binary series: `0=erasure, 1=reception`.
   - Stored in `Data` folder with organized structure
   - **Option 1: BEC Distribution**: Standard erasure channel
   - **Option 2: Gilbert-Elliot**: Two-state Markov channel

4. **Simulation Erasure Series**
   - Pre-determined patterns from data files
   - On-the-fly generation option

### Component Details

#### 1. utils - Configuration Setup

- **config.py** -- Input parameters storage
- **config_setup.py** -- Conversion from JSON (acknowledge when adding new fields to CFG dictionary)

#### 2. run -- Main Running Functions

- **run_1.py** -- Runs one simulation and saves results
- **run_all.py** -- Runs **rep** (repeats) times with different noise realizations (Monte-Carlo)
- **data_storage.py** -- Class to store simulation data
- **plot_results.py** -- Result visualization
- *Set_sim_params -- Not used in this version*

#### 3. ns -- Network System

The actual system implementation - nodes, buffers, packet generation and transmission

#### 4. erasures_generator
These are stand-alone scripts, run by themselves.
- **For Channel Erasure Seris**
  - BEC Channels
  - Gilbert Eliot model

- **For Source Input**
  - Bernulli Distribution
    
- **Format**:
  - Define number of channels
  - Choose model parameters
  - Set main data path
  - Results saved in `"main_path/Type/ch_n/"` (read using `"er_load": "from_csv"`)

#### 5. acrlnc_node

AC-RLNC encoder and decoder operations implementation
