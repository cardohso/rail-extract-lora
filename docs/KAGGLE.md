# Running on the free Kaggle GPU

The notebook is `notebooks/kaggle_run.ipynb`. It clones this public repository, installs the
locked libraries (`uv sync --frozen --group train`), runs one stage and zips the outputs. No token
or key is needed: the base model is public.

1. Sign in at kaggle.com. Internet access in notebooks needs a phone-verified account.
2. Create a new notebook, then **File > Import Notebook** and upload `notebooks/kaggle_run.ipynb`.
3. In the settings panel set **Accelerator: GPU T4 x2** and **Internet: on**.
4. **Save Version > Save & Run All**. You can close the browser while it runs.
5. When it finishes, download `rail_extract_output.zip` from the version's **Output** tab and put
   it in this repository's folder.

Only GPU 0 is used, so timings are for one T4. A stopped run resumes from its last checkpoint only
if `runs/` still exists; Save & Run All starts clean, so a new version starts from step 0.

## If a run fails or you interrupt it

Interrupting a cell stops its command now, but a run that was already started by an older version
of the notebook may still hold the GPU. The notebook refuses to start if the GPU is already busy.
Fix: **Run > Restart Session**, or use **Save & Run All**, which always starts on a clean machine.
Do not run the cells by hand twice in one session.
