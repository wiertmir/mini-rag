To avoid keeping large amount of data in git repository, initially pdf data folders are empty.

To run the application with sample data please run `python download_pdfs.py` first. This would dowload the data based on information in `documents.csv`.

All downlaoded pdf are official IPCC AR6 reports.

All answers in `questions.jsonl` have set `review_status` as 'draft_not_page_verified', they would be later updated with their page and quote references.