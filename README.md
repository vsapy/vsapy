# vsapy - Vector Symbolic Architecture(VSA) library.
This library implements the common methods used in hyperdimensional computing/Vector Symbolic Architectures. Namely
`bind`, `unbind` and some bundling operations, `bag`, `ShiftedBag`, `NamedBag` and a hierarchical bundling 
method `CSPvec`. The main objective of the library is to enable a single API to cater for various flavours of VSA, 
including `Binary Spatter Codes`, `Ternary`, `TernaryZero`, `Laiho` and `LaihoX` (a simplified `Laiho` that 
is faster and supports incremental bundling without catastrophic forgetting). 

A set of demo test cases are supplied to demonstrate the use of the library calls.


## If installing from PyPi simply
  - Poetry add vsapy
<br/>or<br/>
  - pip install vsapy (into your environment)

## Installing from source
  - clone the code to a directory of your choice, say "vsapy"

### Installing Dependancies  
- Poetry: the easiest way is using poetry
  - cd vsapy
  - poetry install
  - poetry shell  (to activate the environment)
  

- pip install vsapy
  - create an environment using your favorite environment manager
  - e.g. conda create -n vsapy39 python=3.9
  - conda activate vsapy39
  - pip install -r requirements.txt

### Usage
Hint: Valid values for `VSaType` are, `VsaType.BSC`, `VsaType.Laiho`, `VsaType.LaihoX`(fastest), `VsaType.Tern`, 
`VsaType.TernZero` and `VsaType.HRR`\
(** Note, the demos listed below will not run with type `VsaType.HRR` **). <br/><br/>


- For examples of using the vsapy library, see the code examples in the ./tests directory. Note there are no 
command-line arguments implemented for the tests at the moment. To change the type of VSA in use, edit the code changing
`vsa_type=VsaType.BSC` as mentioned below. All of the test cases can be run by simply invoking from the command line, 
e.g., `$python cspvec_sequence.py`.



  - `cspvec_sequence.py`: This is the most straightforward demo. Try this first. It demonstrates building a sentence as 
a vsa sequence and stepping forward & backwards. Change `vsa_type=VsaType.BSC` in the code to change the type of VSA
used to build the representation. <br/><br/>
  
  - `build_docs.py`: demonstrates combining large documents into a hierarchical vsa code book. The top-level document 
vector is a high-level semantic representation of the entire document. Change `vsa_type=VsaType.BSC` in the code to 
change the type of VSA used to build the representation. <br/><br/>

    - `load_docs.py`: compares the document vectors built using `build_docs.py` at various levels in the 
document hierarchy. <br/><br/> Change `levels_to_extract = [0, 1]`, `0=top-level document vectors`, `1=Act-level vectors`, 
`2=Scene-level vectors` and so on (Can set to any level, e.g., `levels_to_extract = [2]` would compare only 
Scene-level vectors). <br/><br/>

      - Understanding output names: `OE_=Old English`, `NE_=New English`, `nf_=NoFear Shakespeare`, `tk_=NLTK Shakespeare`,
`og_=Alternate Shakespeare`, `ham=Shakespeare's Hamlet` , `mbeth=Shakespeare's Macbeth`. <br/><br/> 

  - `json2vsa.py`: demonstrates the creation of a VSA vector from an input JSON file and shows a comparison of various
JSONs using VSA. Change `vsa_type=VsaType.BSC` in the code to change the type of VSA used to build the representation.


