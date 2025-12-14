# GenderBias_in_AI_Cover_Letters
Study on whether gender bias (in the form of sociolinguistic differences) are present in AI generated cover letters that are generated from otherwise identical prompts.

Data collection:
There are three python files that will collect the responses(1), analyze them with zero-shot classification(2), and then organize the results(3). All three are run from the terminal with the following commands:

Step 1: Data Collection
  python3 /.../python_files/HT_ML_part1.py
  --api-key "YOUR KEY HERE"
  --out-dir "/.../Desktop"
  --iterations n

Step 2: Analyze with zero shot classification. If analyzing more than one file (prompt response), place the files into a folder before hand and select option 2 when prompted. Follow instructions as prompted
  python3 /.../python_files/HT_ML_part2.py

Step 3: Organize results. Follow instructions as prompted
  python3 /.../python_files/HT_ML_part3.py
