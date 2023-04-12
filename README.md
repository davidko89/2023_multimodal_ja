## Project name: 2023_multimodal_ja
* author: Chanyoung Ko
* date: 2023-04-12
* python version: 3.7.7
* opencv version: 4.7.0

## Objective
- Create JA-based multi-modal classification model for non-ASD vs mild-moderate ASD vs severe ASD

## Project structure
* code
    * src 
    * data
        * raw_data
            1. ija
            2. rja_high
            3. rja_low 
        * proc_data
            1. proc_ija
            2. proc_rja_high
            3. proc_rja_low
    

## Dataset
### [`dataset/participant_information_df.csv`](dataset/participant_information_df.csv)
      
## Labels
* Multi-labels: 0 - non-ASD, 1 - mild-moderate ASD, 2 - severe ASD
* Cut-offs for multi-labels are based on on ADOS CSS or CARS cut-off scores
