# Google Colab Setup & Execution Commands

Follow these steps to run the FedAnilPlus simulation on Google Colab:

### 1. Change Directory to Project Folder
Upload your project folder to Colab and move into it:
```bash
# Example: If you uploaded to Google Drive
# from google.colab import drive
# drive.mount('/content/drive')
# %cd /content/drive/MyDrive/FedAnilPlus
```

### 2. Install Dependencies
Run this in a Colab cell to install all required libraries:
```bash
!pip install scikit-learn scikit-learn-extra bitarray tenseal pycryptodome reportlab
```

### 3. Run the Simulation
Use the following command to start the training with optimized settings:
```bash
!python main.py -nd 100 -max_ncomm 50 -ha 80,10,10 -aio 1 -pow 0 -ko 5 -nm 3 -vh 0.08 -cs 0 -B 64 -mn OARF -iid 0 -lr 0.01 -dtx 1 -le 20
```

### 4. Generate Reports (Optional)
To generate the presentation PDFs on Colab:
```bash
# Move into the documentation folder
%cd Presentation_Docs
!python generate_project_pdf.py
!python generate_individual_member_pdfs.py
%cd ..
```

---
**Note:** Ensure you have selected a **GPU Runtime** in Colab (Runtime > Change runtime type > Hardware accelerator > GPU).
