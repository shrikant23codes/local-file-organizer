# local-file-organizer


## Notes:
- Added langsmith tracing to see the tool calls properly (Make sure while integrating use the proper LANGSMITH_ENDPOINT)
- Things done in Phase 2
```
read filesystem metadata
-> produce structured move plan
-> validate path safety
-> summarize proposed actions
-> perform zero mutations
```
- 

- Phase 3: Dry run of organization_plan output:
```
The dry-run validation of the proposed organization plan is successful with no blockers. Here's a summary of the steps:

                                                        1. **Move** `notes.txt` to `Notes/notes.txt`.
                                                                                                                                  2. **Move** `project_plan.md` to `Documents/project_plan.md`.                                                                                                                  3. **Move** `resume.pdf` to `Documents/resume.pdf`.                                                                                                                            4. **Move** `tax_2024.pdf` to `Documents/tax_2024.pdf`.                                                                                                                        5. **Move** `vacation.jpg` to `Images/vacation.jpg`.
                                                                                                                           The plan is ready for approval, and no changes were applied during the dry-run. The destination folders (Notes, Documents, Images) will be created if they do not already exist. Please note that if these folders contain files with the same names, they may be overwritten.
```

## Issues Seen

- Had to increase the max_tokens to 2000 as LLM output was getting truncated at 800
- Had to update prompt with examples and some more rules.
- 