# How to contribute

## How to get started

Before anything else, please install the git hooks that run automatic scripts during each commit and merge to strip the notebooks of superfluous metadata (and avoid merge conflicts). After cloning the repository, run the following command inside it:
```
nbdev_install_git_hooks
```

## Did you find a bug?

* Ensure the bug was not already reported by searching on GitHub under Issues.
* If you're unable to find an open issue addressing the problem, open a new one. Be sure to include a title and clear description, as much relevant information as possible, and a code sample or an executable test case demonstrating the expected behavior that is not occurring.
* Be sure to add the complete error messages.

#### Did you write a patch that fixes a bug?

* Open a new GitHub pull request with the patch.
* Ensure that your PR includes a test that fails without your patch, and pass with it.
* Ensure the PR description clearly describes the problem and solution. Include the relevant issue number if applicable.

## PR submission guidelines

* Keep each PR focused. While it's more convenient, do not combine several unrelated fixes together. Create as many branches as needing to keep each PR focused.
* Do not mix style changes/fixes with "functional" changes. It's very difficult to review such PRs and it most likely get rejected.
* Do not add/remove vertical whitespace. Preserve the original style of the file you edit as much as you can.
* Do not turn an already submitted PR into your development playground. If after you submitted PR, you discovered that more work is needed - close the PR, do the required work and then submit a new PR. Otherwise each of your commits requires attention from maintainers of the project.
* If, however, you submitted a PR and received a request for changes, you should proceed with commits inside that PR, so that the maintainer can see the incremental fixes and won't need to review the whole PR again. In the exception case where you realize it'll take many many commits to complete the requests, then it's probably best to close the PR, do the work and then submit it again. Use common sense where you'd choose one way over another.

## Do you want to contribute to the documentation?

* Docs are automatically created from the notebooks in the nbs folder.


## Writing a tutorial

Tutorials are held to a written contract: [Tutorial contract](nbs/tutorial_contract.ipynb).
Read it before writing or editing a page — it states the voice, the required sections, the
measurement rules (an accuracy carries its `n`, a percentage that comes from counts is
written with the counts, an interval is recomputed, a latency carries a scope line), and
the rule that every number in prose must be printed by a cell on the same page.

The contract is enforced by the hidden cells of that same notebook, which run in the
default test suite:

```
nbdev_test --path nbs/tutorial_contract.ipynb
```

There is a second, much slower gate that re-executes every tutorial in a copy of its
folder and reports what the code no longer reproduces. It needs the datasets and the
optional dependencies and takes about an hour, so it is behind its own flag:

```
python -c "from nbdev.test import nbdev_test; nbdev_test(path='nbs/tutorial_contract.ipynb', flags='tutorials')"
```

### When the checks fail

* **Fix the page.** That is the expected outcome for anything you just wrote: no new
  violation can be waived.
* **Clearing a quarantine entry.** Pages written before the contract carry known
  violations, listed per page in the `QUARANTINE` dict of the contract notebook. When you
  rewrite such a page, delete its entry in the same commit — the gate fails with
  *"now passes R4: delete the entry"* as soon as the entry stops being used. `QUARANTINE`
  can only shrink: adding a page or a rule that is not already in `_LANDED` fails.
* **Allowing one sentence through the word list.** An honest caveat that happens to contain
  a listed word goes into `LEXICON_ALLOW[page]` verbatim, as a whole sentence, so the diff
  shows exactly what was allowed. It is not a per-page switch.
* **`_LANDED` is edited by review only.** It is the frozen copy of the quarantine taken
  when the contract landed and it is what makes the quarantine one-way; no other change
  should touch it.
