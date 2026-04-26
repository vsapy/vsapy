# vsapy - Vector Symbolic Architecture (VSA) library

⚠️ **Version 0.10+ Update — Breaking Changes**

For a code-structure overview, see [ARCHITECTURE.md](/Users/chrissimpkin/Documents/Nexus/vsapy/ARCHITECTURE.md).

---

## Python Version Requirement (Breaking Change)

As of this release, **vsapy now requires Python >=3.11 and <3.13**.

```toml
python = ">=3.11,<3.13"
```

This restriction allows us to:

- Standardise development across Poetry-managed environments  
- Leverage modern typing and performance improvements  
- Remove legacy compatibility layers  
- Align CI/CD and local development setups  

If you are currently using Python 3.9 or 3.10, you will need to upgrade.

Recommended setup:

```bash
pyenv install 3.11.x
pyenv local 3.11.x
poetry env use 3.11
poetry install
```

---

## NumberLine Refactor & Extensions

This release significantly refactors and extends the **NumberLine** implementations.

### Linear NumberLine Improvements

- Cleaner behaviour across extended ranges  
- Improved quantisation handling  
- More consistent HD similarity gradients  
- Clearer diagnostic plotting support  
- Better geometric interpretation of hyperdimensional distance  

### New Circular Variants

The following circular representations are now included:

- `CircularNumberLineFolded`
- `CircularNumberLineRotational`

These allow:

- True rotational similarity (distance depends only on angular difference)  
- Periodic encodings  
- Geometry-aware embedding behaviour  
- Exploration of circular manifolds in hyperdimensional space  

These representations are particularly useful for:

- Periodic signals  
- Angular encodings  
- Cyclic variables (e.g., time-of-day, phase)  
- Spatial or geometric modelling  
- Investigating rotational invariance in VSA  

---

## Additional & Updated Demos

The demo suite has been expanded and cleaned up:

- Improved hierarchical document demos  
- Expanded JSON bundling examples  
- NumberLine comparison and diagnostic plots  
- Clearer exploration of embedding geometry  

---

# Library Overview

This library implements the common methods used in hyperdimensional computing / Vector Symbolic Architectures. Namely:

- `bind`
- `unbind`
- Bundling operations:
  - `bag`
  - `ShiftedBag`
  - `NamedBag`
- Hierarchical bundling method:
  - `CSPvec`

The main objective of the library is to enable a **single API** to cater for various flavours of VSA, including:

- `Binary Spatter Codes`
- `Ternary`
- `TernaryZero`
- `Laiho`
- `LaihoX` (a simplified Laiho that is faster and supports incremental bundling without catastrophic forgetting)
- `HRR`

A set of demo test cases are supplied to demonstrate the use of the library calls.

---

# Installation

## Install from PyPI

```bash
poetry add vsapy
```

or

```bash
pip install vsapy
```

---

# Installing from Source

Clone the repository:

```bash
git clone <repo-url>
cd vsapy
```

## Installing Dependencies (Poetry Recommended)

```bash
poetry install
poetry shell
```

## Installing with pip

Create and activate a virtual environment:

```bash
conda create -n vsapy python=3.11
conda activate vsapy
pip install -r requirements.txt
```

---

# Usage

Valid values for `VsaType` are:

- `VsaType.BSC`
- `VsaType.Laiho`
- `VsaType.LaihoX` (fastest)
- `VsaType.Tern`
- `VsaType.TernZero`
- `VsaType.HRR`

> **Note:** The demos listed below will not run with `VsaType.HRR`.

For examples of using the vsapy library, see the code examples in the `./tests` directory.

There are currently no command-line arguments implemented for the tests.  
To change the VSA type, edit the code and modify:

```python
vsa_type = VsaType.BSC
```

All test cases can be run directly from the command line, for example:

```bash
python cspvec_sequence.py
```

---

# Demo Files

## `cspvec_sequence.py`

The most straightforward demo.  
Demonstrates building a sentence as a VSA sequence and stepping forward & backwards.

---

## `build_docs.py`

Combines large documents into a hierarchical VSA code book.

The top-level document vector is a high-level semantic representation of the entire document.

---

## `load_docs.py`

Compares document vectors built using `build_docs.py` at various levels in the document hierarchy.

Modify:

```python
levels_to_extract = [0, 1]
```

Where:

- `0` = top-level document vectors  
- `1` = Act-level vectors  
- `2` = Scene-level vectors  
- etc.

You may set to any level (e.g., `[2]` to compare Scene-level only).

Understanding output names:

- `OE_` = Old English  
- `NE_` = New English  
- `nf_` = NoFear Shakespeare  
- `tk_` = NLTK Shakespeare  
- `og_` = Alternate Shakespeare  
- `ham` = Hamlet  
- `mbeth` = Macbeth  

---

## `json2vsa.py`

Demonstrates creation of a VSA vector from an input JSON file and comparison of multiple JSON structures using VSA.
