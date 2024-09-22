# Exit if conda environment garland-composer is not activated.
if [ "$CONDA_DEFAULT_ENV" != "garland-composer" ]; then
    echo "Please activate the garland-composer conda environment."
    echo "Run: conda activate garland-composer"
    exit 1
fi

streamlit run app.py --server.runOnSave=true
