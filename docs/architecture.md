# System Architecture

## Dataset Usage

This project uses two customer support datasets provided in the problem statement.
Each row represents a historical customer support interaction.

The datasets are stored in the `data/` directory and processed in
`core/embeddings.py`, where ticket texts are converted into vector embeddings
and indexed using FAISS. These embeddings are later retrieved during query-time
to generate grounded responses.