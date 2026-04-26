# vsapy Architecture

This document explains how `vsapy` is structured and how its main abstractions fit together.

## Purpose

`vsapy` is a Vector Symbolic Architecture (VSA) / hyperdimensional computing library with two linked goals:

- provide a single API over several different hypervector representations
- support higher-level structured encodings for sequences, documents, and scalar domains

The library is designed so most application code can work in terms of:

- random hypervectors
- `bind` / `unbind`
- bundling
- similarity search

without needing to rewrite logic for each VSA backend.

## Architectural Layers

The codebase is easiest to understand as four layers.

### 1. Typed Hypervector Core

Files:

- [vsapy/vsatype.py](/Users/chrissimpkin/Documents/Nexus/vsapy/vsapy/vsatype.py)
- [vsapy/vsapy.py](/Users/chrissimpkin/Documents/Nexus/vsapy/vsapy/vsapy.py)
- backend implementations in [vsapy/bsc.py](/Users/chrissimpkin/Documents/Nexus/vsapy/vsapy/bsc.py), [vsapy/tern.py](/Users/chrissimpkin/Documents/Nexus/vsapy/vsapy/tern.py), [vsapy/ternzero.py](/Users/chrissimpkin/Documents/Nexus/vsapy/vsapy/ternzero.py), [vsapy/hrr.py](/Users/chrissimpkin/Documents/Nexus/vsapy/vsapy/hrr.py), [vsapy/laiho.py](/Users/chrissimpkin/Documents/Nexus/vsapy/vsapy/laiho.py), and [vsapy/laihox.py](/Users/chrissimpkin/Documents/Nexus/vsapy/vsapy/laihox.py)

Core ideas:

- `VsaType` identifies the active representation.
- `VsaBase` subclasses `numpy.ndarray` and carries VSA metadata such as `vsa_type`.
- Each backend implements the same conceptual operations:
  - `randvec`
  - `normalize`
  - `bind`
  - `unbind`
  - `hdist`
  - `hsim`
- `vsapy.py` exposes the common façade so user code can stay backend-agnostic.

This layer is the algebraic foundation of the library.

### 2. Bundling and Composite Vector Containers

Files:

- [vsapy/bag.py](/Users/chrissimpkin/Documents/Nexus/vsapy/vsapy/bag.py)

Core ideas:

- `PackedVec` stores packed vectors for memory efficiency where that makes sense.
- `BagVec` bundles vectors and tracks how many vectors were combined.
- `ShiftedBag` adds simple order sensitivity by rolling each vector before bundling.
- `NamedBag`, `RawVec`, and `BareChunk` add naming, raw summed vectors, and hierarchy traversal helpers.

This layer turns raw hypervector operations into reusable semantic containers.

### 3. Shared Role and Symbol Space

Files:

- [vsapy/role_vectors.py](/Users/chrissimpkin/Documents/Nexus/vsapy/vsapy/role_vectors.py)

Core ideas:

- `RoleVecs` generates the shared random vectors used as the protocol for structured encodings.
- These include:
  - character/symbol vectors
  - numeric symbol vectors
  - stop vectors
  - sequence metadata tags
  - permutation vectors
  - parent/child role vectors
- `create_role_data()` can rebuild or reload these objects while keeping them timestamp-synchronized.

This layer ensures that independently created structures agree on the same role vocabulary.

### 4. Structured Encoders

Files:

- [vsapy/cspvec.py](/Users/chrissimpkin/Documents/Nexus/vsapy/vsapy/cspvec.py)
- [vsapy/vsa_tokenizer.py](/Users/chrissimpkin/Documents/Nexus/vsapy/vsapy/vsa_tokenizer.py)
- [vsapy/number_line.py](/Users/chrissimpkin/Documents/Nexus/vsapy/vsapy/number_line.py)

Core ideas:

- `CSPvec` builds ordered composite vectors from child vectors.
- `VsaTokenizer` turns words, sentences, and documents into hierarchical VSA structures.
- `NumberLine` and the circular variants map scalar values into hypervectors and decode them back by nearest-neighbour search.

This layer is where the library becomes an application-building toolkit rather than only a set of vector operators.

## Core Abstractions

### `VsaBase`

`VsaBase` is the central type abstraction.

Responsibilities:

- holds the `vsa_type`
- dispatches construction to the right subclass
- preserves backend metadata through ndarray operations
- exposes helper methods such as type validation and truncation to matching lengths

The important design choice here is that vectors remain plain ndarray-like objects while still carrying enough metadata to route operations correctly.

### Backend Implementations

Each backend defines the algebra for one VSA family.

Examples:

- `BSC` uses binary vectors, XOR binding, and majority threshold normalization.
- `Tern` and `TernZero` use signed ternary vectors and multiplicative-style binding.
- `HRR` uses real-valued vectors with circular convolution/correlation.
- `Laiho` and `LaihoX` support sparse slot-style representations with different bundling behavior.

This means a lot of higher-level code can operate over a shared interface while preserving backend-specific semantics.

### `BagVec`

`BagVec` is the basic bundling abstraction.

Responsibilities:

- sum or bundle a collection of vectors
- normalize using the active VSA backend
- keep track of `vec_cnt`, which matters for correct normalization and later interpretation

This vector-count tracking is important because bundling is not just a pure sum; decoding quality depends on knowing how many vectors contributed.

### `RoleVecs`

`RoleVecs` is the source of all shared random role vectors used by the higher-level encoders.

It acts as the coordination layer for:

- symbol alphabets
- sequence positions
- stop/start markers
- metadata tags

Without a shared role-vector space, separately built encodings would not be interoperable.

### `CSPvec`

`CSPvec` is the main ordered-structure encoder.

Conceptually it:

- takes a list of child vectors
- binds each child with position/permutation structure
- adds stop/control information
- bundles everything into a composite vector

It also supports hierarchical chunking, so large lists can be recursively compressed into chunk trees.

This is the main mechanism behind the document and sequence demos.

### `VsaTokenizer`

`VsaTokenizer` is a text-to-VSA adapter built on top of `RoleVecs` and `CSPvec`.

It can:

- build word vectors from character symbols
- optionally reuse cached word vectors
- encode sentences as chunk hierarchies
- optionally project external real-valued embeddings into binary hypervectors via `Real2Binary`

This gives the library both a symbolic composition path and a bridge from conventional embedding models into VSA space.

### Scalar Embeddings

`number_line.py` introduces a more geometric subsystem for scalar data.

It includes:

- `NumberLine` for linear scalar ranges
- `CircularNumberLineFolded` for periodic scalar encoding with folded geometry
- `CircularNumberLinePhaseProjected` for rotationally smoother circular encoding using phase embeddings projected into BSC space

These classes show that the library is not limited to symbolic strings and sequences; it also supports structured continuous domains.

## Data Flow Through The Library

A typical flow looks like this:

1. Choose a backend with `VsaType`.
2. Create role/symbol vectors with `create_role_data()`.
3. Generate atomic hypervectors with `randvec()` or from the symbol dictionary.
4. Compose them with `bind()`, `unbind()`, rolling, and bundling.
5. Wrap them in containers such as `BagVec` or `CSPvec`.
6. Search, decode, or traverse them using similarity and unbinding.

For text, the flow is usually:

1. `RoleVecs` creates symbol vectors.
2. `VsaTokenizer` converts characters to word vectors.
3. Words become `CSPvec` chunks.
4. Sentences and documents become hierarchical chunk trees.
5. Retrieval is performed by similarity search over chunk vectors plus stepwise unbinding for sequence traversal.

For scalar embeddings, the flow is:

1. Build a bank of level vectors.
2. Bind with a domain-separating key if needed.
3. Decode by unbinding and nearest-neighbour search over the bank.

## What Makes The Design Distinct

The key design pattern in `vsapy` is:

`backend-specific vector algebra -> shared API -> reusable structured encoders`

That gives the project a few useful properties:

- experiments can switch between VSA backends with limited application code changes
- higher-level encoders stay focused on structure rather than low-level algebra
- the same library can model symbols, sequences, hierarchies, and scalar domains

In practice, this makes `vsapy` more than a collection of vector ops. It is a framework for constructing recoverable, compositional representations in hypervector space.

## Important Practical Constraints

Some higher-level structures are backend-specific.

Examples:

- `NumberLine` currently targets `BSC`, `Tern`, and `TernZero`.
- circular number line variants currently focus on `BSC`
- the README notes that some demos do not support `HRR`

So the architecture is shared, but not every feature is implemented for every backend.

## Suggested Mental Model

When reading or extending the library, it helps to think in this order:

1. What atomic vectors exist in this encoding?
2. What algebra does the selected backend use?
3. How are those atoms composed: bind, roll, bag, or hierarchical chunking?
4. How will the representation later be decoded or searched?

That mental model matches the codebase well and explains why the repository is organized around both low-level vector math and higher-level semantic encoders.
