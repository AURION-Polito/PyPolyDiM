# PyPolyDiM

This Python code provides a collection of examples demonstrating how to use *PyPolyDim*, the Python binding of the C++ **PolyDiM&GeDiM** 

More details about this librarty can be found at **[PolyDiM website](https://your-website-url.com)**.

## Installation

1. Open a terminal.
2. Create a Python virtual environment.

```bash
python3 -m venv .venv
```

3. Activate the virtual environment.

```bash
source .venv/bin/activate
```

4. Install the required dependencies.

```bash
pip install -r requirements.txt
```

5. Deactivate the virtual environment.

```bash
deactivate
```

# How to use

Run one of the provided examples. Each example can be identified by the *main* keyword. The examples demonstrate how to solve elliptic PDE problems on polytopal meshes using different numerical methods, including FEM, VEM and Z-FEM. 

You are also encouraged to adapt and implement other types of problems. A fairly comprehensive, though not exhaustive, list of possible implementations can be found both in the C++ code and in the reference paper.

For a detailed description of the methods and implementation, please refer to the following paper:

*Stefano Berrone, Andrea Borio, Gioana Teora, and Fabio Vicini, **POLYDIM: A C++ library for POLYtopal DIscretization Methods**, Computer Physics Communications, Volume 320, 2026, 109937, ISSN 0010-4655, [DOI: 10.1016/j.cpc.2025.109937](https://doi.org/10.1016/j.cpc.2025.109937).*

*Cite this paper to use the library.*

## Manteiners

- Gioana Teora, Politecnico di Torino (gioana.teora@polito.it)
- Fabio Vicini, Politecnico di Torino (fabio.vicini@polito.it)





