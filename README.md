# CATNIP_draft
This is the draft version to set up the public facing CATNIP tool. Currently, I'm making a tutorial for the tool.

Part 1 - setting up your Google sheet

Download the Google Sheet template here:

https://docs.google.com/spreadsheets/d/1ZXnlReVCGW3vZHHM23J9_VRH-fQ7u1lA7Zm_0FPPa9k/copy

Part 2 - setting up the code

Create and activate a new environment:

Option 1
```
python3 -m venv myenv

source myenv/bin/activate
```

Option 2
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

Open a Jupyter Lab/notebook
Enter folder called CATNIP_draft
Enter FIGG-CATNIP
Enter Notebook for FIGG-CATNIP_v1_0
There is the tutorial

Now work through and add tests and such
