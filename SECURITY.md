# Security policy

## Supported versions

LangScope has not had a release yet. Security fixes go to the `main` branch; please check that a problem still exists there before reporting it.

| Version | Supported |
|---|---|
| `main` | Yes |
| Anything older | No |

## Reporting a vulnerability

Please report vulnerabilities privately, not in a public issue, pull request or discussion.

Use GitHub's private vulnerability reporting: open the repository's **Security** tab and choose **Report a vulnerability**, or go straight to [the new advisory form](https://github.com/MohamedxChaouachi/LangScope/security/advisories/new).

Include what you can of:

- what the problem is and what an attacker could do with it
- the steps or a proof of concept to reproduce it
- the LangScope commit, Windows version, Python version, and the PyTorch, EasyOCR and Argos Translate versions
- any fix you would suggest

You should get an acknowledgement within 7 days. Once the report is confirmed, a fix will be prepared, and the advisory published after it is available. You will be credited in the advisory unless you would rather not be.

## Scope

LangScope runs locally and needs no account or network service after setup. These are the areas most worth a look:

- **Model and language-pack downloads.** On first use, EasyOCR models and Argos Translate packs are downloaded and then loaded from disk. Problems in how they are fetched, verified or loaded are in scope.
- **Screen contents.** LangScope reads pixels from the region you select and keeps them in memory only. Anything that writes captured screen contents or recognized text somewhere it should not, including logs, is in scope.
- **Dependencies.** A vulnerability in PyTorch, EasyOCR, Argos Translate, PyQt6 or another dependency should go to that project. If LangScope uses it in a way that makes the problem exploitable, report it here too.
