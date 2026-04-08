# Chat History

Running log of prompt-driven changes and local commits.

## 2026-04-08 06:03:19 CDT
- Prompt: Let's add a perminate action:  After every code change do a local git commit with the prompt input as the commit.  Add a log file with chat history as to keep a running log of the process.  This will allow us to have a history of the interactions and be able to revert later as needed.
- Branch: master
- Files:
  - README.md
  - logs/chat_history.md
  - scripts/prompt_commit.sh

## 2026-04-08 06:11:22 CDT
- Prompt: I think we lost the progress bar when downloading resources.  Can we get those back please?
- Branch: master
- Files:
  - bootstrap_assets.py
