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

## System Architecture

### Network Model

![Network Model](https://raw.githubusercontent.com/username/blank-space/main/docs/images/system_overview.png)

1. **N-node network**
   - Data flows between every two adjacent nodes
   - Multicast capability:
       - Intermediate nodes can decode information packets as well as semi-decode.
       - This happens anyway and seen in the respected metrics.
       - This does not affect transmission flow or the semi-decoding process at all.

2. **Slotted Communication**
   - Time horizon: **T** slots
   - One operation per time slot

3. **RTT Handling**
   - All channels use the same local RTT
   - **Important**: In this version AC-RLNC doesn't support heterogeneous RTTs
   - Each packet generation takes 1 time slot
   - For N nodes with local RTT₍ₗ₎, the effective RTT is RTT₍ₗ₎ + 2
     (1 slot for forward packet + 1 slot for feedback)

### Channel Models

1. **Forward Erasure Channels**
   - Erasure rate ε₍ₙ₎ for each channel
   - Channel count determined by `er_rates` array length
   - Variable erasure rates supported via:
     ```
     "er_var_ind": [1],  # Channel index to vary
     "er_var_values": [0.2, 0.5, 4],  # [start, end, steps]
     ```
     **Note**: This overwrites the epsilon in `er_rates`

2. **Gilbert-Elliot Channel Option**
   - Supports one GE channel with fixed parameters
   - Configure in `er_type` parameter:
     ```
     "er_type": "BEC"  # Options: BEC, GE_p_0.01_g_0.1_b_1, etc.
     "ge_channel": [1]  # GE channel index (ignored if BEC)
     ```

3. **Source Packet Generation**
   - **Option 1**: Continuous generation (one packet per time slot)
     - Configured with `"in_type": "all"`
   - **Option 2**: Bernoulli distribution
     - Configured with `"in_type": "ber"`
     - Uses pre-generated series from Data folder

4. **Feedback Channels**
   - Assumed to be reliable.

## Packet Structure and Data Representation

### Packet Representation

Blank Space uses an index-based representation rather than actual data transfer:

1. Each packet is represented by a local index at each node
2. Source node indices represent the **information packets** 
3. Coded packets are represented by window bounds [w_min, w_max]:

   | Field | Description |
   |-------|-------------|
   | w_min | First information packet index in the combination |
   | w_max | Last information packet index in the combination |

4. The coding window evolves based on:
   - w_min: incremented based on decoded packet feedback
   - w_max: determined by the encoder algorithm
   - Buffer may contain packets with indices > w_max

### Packet Fields

Each packet contains essential information:

```
Packet {
    nc_header: [w_min^(n-1), w_max^(n-1)],  // Previous node indices
    nc_id: local_id,                        // Local identifier
    src: prev_node,                         // Source node
    FEC_Type: transmission_type             // Packet type
}
```

### FEC Types

| Type | Description |
|------|-------------|
| NEW | New information packet - a new **local w_max** in the packet |
| FEC | Forward Error Correction packet from the a-prior FEC mechanism |
| FB-FEC | Forward Error Correction packet from the posterior FEC mechanism (FB=Feedback) |
| EOW | Forward Error Correction packet from the End of Window mechanism |
| EMPTY-BUFFER | Pause transmission from the No-New No-FEC mechanism (EMPTY=Pause transmission) |
| EMPTY-BS |  Pause transmission from the Blank Space mechanism |
   
### Node Operations

1. **Index Handling**:
   - Semi-decoding: Uses previous node indices
   - Encoding: Uses current node indices
   
2. **Node Implementation**:
   -2.	To have all encoders implemented in the same manner, the source and destination are implemented by an extra naïve node.
   - All intermediate nodes use identical encoder/decoder structure (ac_node\ac_mix_all+encoder\encoder_mix_all).

## Network Coding Implementation

### Prediction Methods

The algorithm supports various noise/erasure prediction methods:

| Method | Description |
|--------|-------------|
| `genie` | Knows the true erasure rate but not actual realizations |
| `oracle` | Knows the actual erasure realization (theoretical bound) |
| `stat` | Statistical estimation (mean over feedback) |
| `stat_max` | Mean + standard deviation (experimental) |

Configure via: `"er_estimate_type": "stat"`

### Node Implementation

1. **ns/nc_enc.py**
   - Connects general code with encoder
   - Handles result storage

2. **acrlnc_node Modules**:

   - **eps_est**
     - Maintains feedback history
     - Estimates erasure rate
     - Configurable prediction method

   - **encoder.py + ac_node.py**
     - Full NET AC-RLNC implementation at each node
   
   - **encoder_mix_all.py + ac_node_mix_all.py**
     - Simplified version with:
       - AC-RLNC at source only (H₀)
       - Intermediate nodes perform full packet mixing







