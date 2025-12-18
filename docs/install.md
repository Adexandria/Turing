# Install
This project requires **Python 3.12**. Please ensure that this version is installed before continuing.

!!! note  
    The application may not run correctly with earlier Python versions.

To check your current Python version, run:

``` bash
python --version
``` 

or

```bash
python3 --version
``` 

If you do not have Python 3.12 installed, you can download it from the official website: [https://www.python.org/downloads/release/python-3120/](https://www.python.org/downloads/release/python-3120/)


----------

## Clone the Repository

Use Git to download the project:

``` bash
git clone https://github.com/se4ai2526-uniba/Turing.git
```

----------

## Set Up the Environment

It is recommended to use a virtual environment to manage dependencies.

### Create a Virtual Environment

```bash
python -m venv venv
```

### Activate the Virtual Environment

=== "Windows"

    ``` bash
    venv\Scripts\activate
    ```

=== "Linux/macOS"

    ``` bash
    source venv/bin/activate
    ```
!!! tip  
    If the environment fails to activate, ensure you have permission to execute scripts on your system.

----------

## Install Dependencies

Install the required packages using:

```bash
pip install -e pyproject.toml
```

!!! warning  
    Ensure the virtual environment is activated before installing dependencies, or the packages may be installed globally.


### GPU Support
The model is capable of running on a **GPU**, which can significantly speed up training and inference.  
To enable GPU acceleration, you need to install a CUDA-compatible version of PyTorch.

Run the following command to install PyTorch with **CUDA 13.0** support:

``` bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

If you are using a different CUDA version, you can find the correct installation command by visiting the official PyTorch installation guide:

**PyTorch Installation Guide:** [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Before installing, ensure that:

-   Your system has a supported **NVIDIA GPU**
    
-   The correct **CUDA toolkit** is installed
    
-   Your **Python environment** (virtual environment recommended) is active
    

Using the right CUDA version ensures that PyTorch can fully utilize your GPU for faster and more efficient computation.