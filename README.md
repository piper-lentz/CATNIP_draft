# CATNIP_draft
This is the draft version to set up the public facing CATNIP tool. Currently, I'm making a tutorial for the tool.

Part 1 - setting up your Google sheet

Download the Google Sheet template here:

https://docs.google.com/spreadsheets/d/1ZXnlReVCGW3vZHHM23J9_VRH-fQ7u1lA7Zm_0FPPa9k/copy

This sheet comes preloaded with information for NIR and ALMA images for the disks EM* AS 209 and HD135344B. The files for these disk images are uploaded in the folder as a part of this tutotrial.

Part 2 - setting up the code

Create and activate a new environment:

```
conda create --name myenv python=3.12

conda activate myenv
```

How to download:
```
git clone https://github.com/piper-lentz/CATNIP_draft.git
```

cd into CATNIP_draft folder

Install dependancies:
```
pip install -r requirements.txt
```

Open a Jupyter Lab
```
conda install jupyterlab

jupyter lab
```

and 

Now work through and add tests and such
