# Step-by-step GitHub and DOI setup

This local folder is intentionally not a Git repository. Complete the steps
below after reviewing the files.

## 1. Final local review

1. Confirm the BSD 3-Clause license and Bofei Xu copyright notice in `LICENSE`.
2. Confirm author names and affiliations in `CITATION.cff`.
3. Run the tests and the synthetic example notebook in the pinned environment.
4. Check `SHA256SUMS.txt` after all final artifact changes. Regenerate it if any
   model, scaler, metadata, or result file changes.
5. Confirm that no raw personal-communication record is present.

## 2. Create an empty GitHub repository

On GitHub, create a repository such as `liquefaction-onset-gru-aic`. Do not ask
GitHub to add a README, `.gitignore`, or license because this folder already
contains those files. It is reasonable to keep the repository private during
final verification and make it public immediately before archival.

## 3. Initialize Git and Git LFS locally

The five model files total approximately 475 MiB, and each is close to the
ordinary GitHub file-size limit. Use Git LFS. From this folder in PowerShell:

```powershell
git init -b main
git lfs install
git lfs track "*.keras"
git add .gitattributes
git add .
git lfs ls-files
git status
git commit -m "Initial public release"
```

The included `.gitattributes` already declares the `.keras` rule; running
`git lfs track` confirms it in the local Git configuration. Verify that all
five `best_model.keras` files appear in `git lfs ls-files` before pushing.

GitHub's current guidance is:

- <https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage>
- <https://docs.github.com/en/repositories/working-with-files/managing-large-files/configuring-git-large-file-storage>

## 4. Connect and push

Replace the placeholders with the repository URL shown by GitHub:

```powershell
git remote add origin https://github.com/OWNER/liquefaction-onset-gru-aic.git
git push -u origin main
```

Clone the repository into a separate temporary location and run
`git lfs pull` as a download check. Confirm that the five model files are real
HDF5 files, not small LFS pointer text files.

## 5. Public-release checks

Before changing visibility to public:

1. Read the rendered README on GitHub.
2. Confirm that GitHub recognizes `CITATION.cff`.
3. Confirm the chosen license is displayed.
4. Confirm no restricted record or local absolute path is present.
5. Run the notebook from the clean clone.
6. Under repository **Settings > Archives**, enable **Include Git LFS objects
   in archives**. Otherwise GitHub-generated ZIP and tar archives contain only
   pointer files in place of the five trained models.

## 6. Obtain the persistent DOI through Zenodo

The most reliable route for this repository is a manual Zenodo Software
upload, because it guarantees that the DOI deposit contains the complete Git
LFS model objects rather than pointer files.

1. Tag the verified GitHub commit as `v1.0.0` and create the corresponding
   GitHub release.
2. From the repository root, create a ZIP from the complete working tree while
   excluding local Git and generated-output directories:

   ```powershell
   tar --exclude=.git --exclude=.venv --exclude=outputs --exclude=notebook_outputs `
       -a -cf ..\liquefaction-onset-gru-aic-v1.0.0.zip .
   ```

3. Inspect or extract the ZIP and confirm that it contains `LICENSE`,
   `.gitattributes`, and five `.keras` files of approximately 95 MiB each.
4. In Zenodo, create a **New upload** with resource type **Software**. Upload
   the verified ZIP; use version `1.0.0`, license `BSD-3-Clause`, and add the
   GitHub repository URL as a related identifier.
5. Use Zenodo's **Get a DOI now!** option if the DOI must be inserted into the
   manuscript before publishing the deposit. If the DOI is added to repository
   files, rebuild and replace the ZIP before publishing the Zenodo record.
6. Verify the title, creators, version, license, related GitHub URL, file size,
   and open-access state, then publish the Zenodo record.
7. Download the published ZIP and again verify that it contains the five
   complete `.keras` models. Do not cite or place the DOI in the manuscript
   until this check passes.

The automatic Zenodo-GitHub integration is also usable, but only after
enabling Git LFS objects in GitHub archives and confirming that the resulting
Zenodo archive contains the complete models.

Official Zenodo guidance:

- <https://help.zenodo.org/docs/github/enable-repository/>
- <https://help.zenodo.org/docs/github/archive-software/github-upload/>
- <https://help.zenodo.org/docs/github/describe-software/citation-file/>
- <https://help.zenodo.org/docs/deposit/create-new-upload/>

## 7. Final manuscript update

Replace the repository DOI placeholder in the manuscript's Data availability
section with the DOI for the archived public release. Use the DOI URL, not only
the mutable GitHub URL. If desired, add the concept DOI or release DOI badge to
the README in a later small release.

