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

## Issues Seen

- Had to increase the max_tokens to 2000 as LLM output was getting truncated at 800
- Had to update prompt with examples and some more rules.
- 