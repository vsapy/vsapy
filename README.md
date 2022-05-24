# vsapy - Vector Symbolic Architecture(VSA) library.

## Getting Started
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
NOTE: Demos listed below will not run with type `VsaType.HRR`.


- For examples of using the vsapy library see the code examples in the ./tests directory. Note there are no comamd-line
arguments implemented for the tests at the moment. To change the type of VSA in use edit the code changing
`vsa_type=VsaType.BSC` as mentioned below. All of the tescases can be run by simply invoking from the command line 
e.g., `$python cspvec_sequence.py`.



  - `cspvec_sequence.py`: This is the simplest demo, try this first. It demonstrates building a sentence as a vsa 
sequence and steping forwarsd & backwards. Change `vsa_type=VsaType.BSC` in the code to change the type of VSA in use
to build the represntation. <br/><br/>
  
  - `build_docs.py`: demonstrates combining large documents into a hierarchical vsa code book. The top level document 
vector is a high-level semantic representation of the entire document. Change `vsa_type=VsaType.BSC` in the code to 
change the type of VSA in use to build the represntation. <br/><br/>

  - `load_docs.py`: demonstrates comparing the document vectors built using `build_docs.py` at various levels in the 
document hierarchy. <br/><br/> Change `levels_to_extract = [0, 1]`, `0=top-level document vectors`, `1=Act-level vectors`, 
`2=Scene-level vectors` and so on (Can set to any level, e.g., `levels_to_extract = [2]` would compare only 
Scene-level vectors). <br/><br/>

  - `json2vsa.py`: demonstrates creation of a VSA vector from an input JSON file and shows comparison of various JSONs
using VSA. Change `vsa_type=VsaType.BSC` in the code to change the type of VSA in use to build the represntation.


