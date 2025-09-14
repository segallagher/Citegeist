@echo off
REM Check if vectorstore directory exists
IF NOT EXIST "%VECTORSTORE_DIR%" (
    echo %VECTORSTORE_DIR% not found

    REM Check if dataset file exists
    IF NOT EXIST "%DATASET_PATH%" (
        echo %DATASET_PATH% not found

        REM Check if paper directory exists
        IF NOT EXIST "%PAPER_DIR%" (
            echo %PAPER_DIR% not found, no papers, cannot create vectorstore
            echo Please run the scraper or download paper collection
            exit /b 1
        )

        REM Papers found but no dataset
        echo Assembling dataset
        python dataset_creation\assemble_dataset.py
    )

    echo Generating vectorstore
    python create_vectorstore_db.py
)
