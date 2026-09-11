# Data

The original development run used Parquet source and processed tables totaling more than 80 MB. They are intentionally omitted from the lightweight GitHub-ready package to keep the repository practical.

Use the reproducible generator in `src/data_generation/` to recreate the source tables, then run the feature/model pipeline documented in the root README.

Published model and policy results are preserved in `src/models/artifacts/`, and Power BI-ready analytical exports are preserved in `dashboards/export/`.
