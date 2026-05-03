# Analyzing Vision Zero Outcomes in Collisions, Community, and Policing

## Project Overview

In 2014, San Francisco adopted Vision Zero, a 10-year plan focused on eliminating fatal and severe crashes. While this initiative was designed to address collisions, enforcement of its subsequent policies may have also affected daily traffic activity and oversight. Currently there is limited analysis and documentation available to the public regarding the reform’s impact on traffic activity. This project will examine policy effectiveness by comparing pre- and post implementation outcomes associated with Vision Zero in San Francisco. The scope of this project defines ”effectiveness” through two key metrics: 1. impact on traffic outcomes 2. changes in policing behavior across demographic groups.

The impact of Vision Zero San Francisco on traffic outcomes is centered around examining whether the changes reduced traffic-related fatalities and injuries. Our accompanying analysis of potential bias in policing behavior after Vision Zero focuses on identifying changes in traffic-enforcement patterns and assessing whether these changes have disproportionately affected marginalized communities. A quantitative assessment of these objectives provides insight into whether Vision Zero, given the ethical implications identified in this study, is suitable for broader adoption across U.S. cities.

## Data Description

A description of the relevant data is provided within each subdirectory.

## Installation Instructions

To clone this repository to your local machine: 
1.  Download [git](https://git-scm.com/). 
2.  Open your Terminal/Command Prompt application. 
3.  Ensure Git-LFS is installed for accessing large files. Enter the following prompt:

```         
git lfs version
```

If output indicates `'lfs' is not a git command`, install it via the [website](https://git-lfs.com/) or the following steps:

```         
brew install git-lfs       # for macOS
apt-get install git-lfs    # Linux
```

Initalize it via:

```         
git lfs install
```

4.  Change your working directory to the local folder where you want the repository to be saved (change `local_folder_path` to the actual path name).

```         
cd local_folder_path
```

5.  Clone the github repository.

```         
git clone https://github.com/utexas-SDS-357/sds357-project-sp26-con-founders.git
```

A new directory named after the repository will be created, containing all files, history, and metadata. By default, the main branch will be visible and contain the following files:

| Directory/File | Description |
|------------------------------------|------------------------------------|
| `bias` | Data, source code, and output for the Bias objective |
| `did` | Data, source code, and output for the Impact objective |
| `.gitattributes` | File used by Git to define how specific files and paths should be handled, used for large file storage |
| `.gitignore` | File used by Git to prevent tracking unnecessary files |
| `requirements.txt` | Necessary dependencies to install |

After cloning the repository to your local machine, create and activate a virtual environment to install the necessary project-specific dependencies. 

1.  Change your working directory to the local repository. 
2.  Create a virtual environment:

```         
python3 -m venv .venv
```

3.  Activate the virtual environment: For Mac and Linux users, enter:

```         
source .venv/bin/activate
```

For Windows users,

```         
.venv\Scripts\activate 
```

To install necessary dependencies,

```         
pip install -r requirements.txt 
```

## Usage Instructions

To reproduce any results or make updates to the pipeline, use the installation instructions above to access the relevant files. Code can be accessed and edited using a shell or integrated development environment (e.g. VSCode) that supports Python.

### Dependencies

Necessary dependencies are included in the `requirements.txt` file in the top-level directory.
