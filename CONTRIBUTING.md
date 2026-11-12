# Contributing to LangScope

Thanks for helping. Bug reports, fixes, new OCR or translation engines and documentation are all welcome.

## Before you start

- **Bugs and ideas:** open an issue with the bug report or feature request template. For anything larger than a small fix, open an issue first, so the approach can be agreed before you write the code.
- **Security problems:** do not open a public issue. Follow [SECURITY.md](SECURITY.md).
- Everyone taking part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Setting up

LangScope supports Windows and Python 3.11 or newer. Install PyTorch first (see [Installation](README.md#installation) for the CPU and CUDA builds), then the package in editable mode with the development tools:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

To run only the tests, without PyTorch, EasyOCR or Argos Translate, install `requirements-ci.txt` instead. Tests that need those packages skip themselves with a reason.

## Making a change

1. Branch from `main`.
2. Keep each commit to one logical change, and add or update a test for every behaviour change or bug fix. A bug fix should come with a test that fails without it.
3. Run the same checks as CI, and make sure they all pass:

   ```powershell
   ruff check .
   ruff format --check .
   pytest
   pyright
   ```

4. Update the README when you change behaviour, configuration or a documented number.

### Commit messages

The history uses [Conventional Commits](https://www.conventionalcommits.org/): `type(scope): summary`, for example `fix(overlay): clamp text to the native-pixel surface`. Common types are `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `build`, `ci` and `chore`. Write the summary in the imperative, and use the body to say why the change is needed.

### Code style

- Ruff handles linting and formatting, and Pyright the type checking; the configuration is in `pyproject.toml`.
- Annotate new functions.
- The UI and the workers talk through Qt signals or immutable snapshots. Do not touch a worker's state from another thread.
- New OCR or translation engines go behind the `BaseOCR` and `BaseTranslator` interfaces, so the rest of the pipeline does not change.

## Pull requests

Open the pull request against `main` and fill in the template. CI runs Ruff, builds and checks the wheel, and runs pytest and Pyright on Windows with Python 3.11 and 3.14; it has to pass before a merge. Keep pull requests focused: two unrelated changes are easier to review as two pull requests.

By contributing, you agree that your contribution is licensed under the [MIT License](LICENSE).
