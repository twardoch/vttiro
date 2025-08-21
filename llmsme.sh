#!/usr/bin/env bash
llms . "CHANGELOG.md,WORK.md,CLAUDE.md,PLAN.md,AGENTS.md,.cursorrules,GEMINI.md,plan,TLDR.md,llms.txt"
mv llms.txt issues/
echo "---" >> issues/llms.txt
fd >> issues/llms.txt


