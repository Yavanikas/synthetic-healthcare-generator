from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

def train_and_generate(df):
    # Create metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)

    # Create model
    model = CTGANSynthesizer(metadata, epochs=5)

    # Train model
    model.fit(df)

    # Generate synthetic data
    synthetic = model.sample(len(df))

    return synthetic