import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, model_selection
import matplotlib.pyplot as plt
import time

# Import StellarGraph for GCN implementation
try:
    import stellargraph as sg
    from stellargraph import StellarGraph
    from stellargraph.mapper import FullBatchNodeGenerator
    from stellargraph.layer import GCN
except ImportError:
    print("StellarGraph is not installed. Install it using: pip install stellargraph")

class VHydroGCN:
    """
    Graph Convolutional Network model for hydrocarbon potential prediction
    """
    def __init__(self, node_data, edge_data, facies_segments, 
                 train_size=0.8, validation_size=0.1, random_state=42):
        """
        Initialize the GCN model
        
        Parameters:
        -----------
        node_data : pandas.DataFrame
            DataFrame containing node connections with 'source' and 'target' columns
        edge_data : pandas.DataFrame
            DataFrame containing edge data with 'DEPTH' and 'PET_label' columns
        facies_segments : list
            List of lists containing indices of continuous segments
        train_size : float
            Proportion of data to use for training
        validation_size : float
            Proportion of data to use for validation
        random_state : int
            Random seed for reproducibility
        """
        self.node_data = node_data
        self.edge_data = edge_data
        self.facies_segments = facies_segments
        self.train_size = train_size
        self.validation_size = validation_size
        self.random_state = random_state
        
        # Initialize attributes
        self.graph = None
        self.model = None
        self.generator = None
        self.train_subjects = None
        self.val_subjects = None
        self.test_subjects = None
        self.train_targets = None
        self.val_targets = None
        self.test_targets = None
        self.history = None
        self.target_encoding = None
        
        # Prepare the data
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data for the GCN model"""
        # Create node features DataFrame
        node_features = self.edge_data.drop(columns=['Values', 'PET_label'])
        node_features.set_index('DEPTH', inplace=True)
        
        # Create a StellarGraph
        self.graph = StellarGraph(
            {"paper": node_features}, 
            {"cites": self.node_data[['source', 'target']]}
        )
        
        # Get node labels
        node_subjects = self.edge_data.set_index('DEPTH')['PET_label']
        
        # Split into train, validation, test sets
        train_val_subjects, self.test_subjects = model_selection.train_test_split(
            node_subjects, 
            train_size=self.train_size + self.validation_size,
            test_size=1 - (self.train_size + self.validation_size),
            stratify=node_subjects,
            random_state=self.random_state
        )
        
        # Further split train_val into train and validation
        relative_val_size = self.validation_size / (self.train_size + self.validation_size)
        self.train_subjects, self.val_subjects = model_selection.train_test_split(
            train_val_subjects,
            train_size=1 - relative_val_size,
            test_size=relative_val_size,
            stratify=train_val_subjects,
            random_state=self.random_state
        )
        
        # Encode the labels
        self.target_encoding = preprocessing.LabelBinarizer()
        self.train_targets = self.target_encoding.fit_transform(self.train_subjects)
        self.val_targets = self.target_encoding.transform(self.val_subjects)
        self.test_targets = self.target_encoding.transform(self.test_subjects)
        
        # Create a node generator
        self.generator = FullBatchNodeGenerator(self.graph, method="gcn")
        
        print(f"Train set: {len(self.train_subjects)} nodes")
        print(f"Validation set: {len(self.val_subjects)} nodes")
        print(f"Test set: {len(self.test_subjects)} nodes")
    
    def build_model(self, layer_sizes=[16, 16], activations=["relu", "relu"], 
                    dropout=0.5, learning_rate=0.01):
        """
        Build the GCN model
        
        Parameters:
        -----------
        layer_sizes : list
            List of hidden layer sizes
        activations : list
            List of activation functions for each layer
        dropout : float
            Dropout rate
        learning_rate : float
            Learning rate for Adam optimizer
        """
        # Create GCN model
        gcn = GCN(
            layer_sizes=layer_sizes,
            activations=activations,
            generator=self.generator,
            dropout=dropout
        )
        
        # Create input and output tensors
        x_inp, x_out = gcn.in_out_tensors()
        
        # Add final prediction layer
        predictions = layers.Dense(units=self.train_targets.shape[1], activation="softmax")(x_out)
        
        # Create model
        self.model = Model(inputs=x_inp, outputs=predictions)
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss=losses.categorical_crossentropy,
            metrics=["acc"],
        )
        
        # Print model summary
        print(self.model.summary())
    
    def train(self, epochs=200, patience=50, verbose=1):
        """
        Train the GCN model
        
        Parameters:
        -----------
        epochs : int
            Number of training epochs
        patience : int
            Patience for early stopping
        verbose : int
            Verbosity level (0, 1, or 2)
            
        Returns:
        --------
        pandas.DataFrame
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Create data generators
        train_gen = self.generator.flow(self.train_subjects.index, self.train_targets)
        val_gen = self.generator.flow(self.val_subjects.index, self.val_targets)
        
        # Create early stopping callback
        es_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_acc", 
            patience=patience, 
            restore_best_weights=True
        )
        
        # Train the model
        start_time = time.time()
        self.history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            verbose=verbose,
            shuffle=False,
            callbacks=[es_callback],
        )
        end_time = time.time()
        
        training_time = end_time - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Convert history to DataFrame
        history_df = pd.DataFrame(self.history.history)
        
        return history_df
    
    def evaluate(self):
        """
        Evaluate the model on the test set
        
        Returns:
        --------
        tuple
            (loss, accuracy)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Create test generator
        test_gen = self.generator.flow(self.test_subjects.index, self.test_targets)
        
        # Evaluate on test set
        test_metrics = self.model.evaluate(test_gen)
        
        # Print metrics
        print("\nTest Set Metrics:")
        for name, val in zip(self.model.metrics_names, test_metrics):
            print(f"\t{name}: {val:.4f}")
        
        return test_metrics
    
    def predict(self):
        """
        Generate predictions for all nodes
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with true and predicted labels
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Create generator for all nodes
        all_nodes = self.edge_data['DEPTH']
        all_gen = self.generator.flow(all_nodes)
        
        # Generate predictions
        all_predictions = self.model.predict(all_gen)
        
        # Convert predictions to labels
        node_predictions = self.target_encoding.inverse_transform(all_predictions)
        
        # Create DataFrame with predictions
        predictions_df = pd.DataFrame({
            'DEPTH': all_nodes,
            'PET_label': self.edge_data['PET_label'],
            'Predicted_PET': node_predictions
        })
        
        return predictions_df
    
    def plot_training_history(self):
        """
        Plot training and validation metrics
        
        Returns:
        --------
        matplotlib.figure.Figure
            Training history plot
        """
        if self.history is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Convert history to DataFrame
        history_df = pd.DataFrame(self.history.history)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot loss
        ax1.plot(history_df['loss'], 'b-', label='train')
        ax1.plot(history_df['val_loss'], 'r-', label='validation')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot accuracy
        ax2.plot(history_df['acc'], 'b-', label='train')
        ax2.plot(history_df['val_acc'], 'r-', label='validation')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        
        return fig
    
    def plot_confusion_matrix(self, predictions_df=None):
        """
        Plot confusion matrix of model predictions
        
        Parameters:
        -----------
        predictions_df : pandas.DataFrame, optional
            DataFrame with true and predicted labels.
            If None, predictions will be generated.
            
        Returns:
        --------
        matplotlib.figure.Figure
            Confusion matrix plot
        """
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        
        # Get predictions if not provided
        if predictions_df is None:
            predictions_df = self.predict()
        
        # Create confusion matrix
        cm = confusion_matrix(
            predictions_df['PET_label'],
            predictions_df['Predicted_PET'],
            labels=self.target_encoding.classes_
        )
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot confusion matrix
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                         xticklabels=self.target_encoding.classes_,
                         yticklabels=self.target_encoding.classes_)
        
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        return plt.gcf()

def create_and_train_gcn_model(node_data, edge_data, facies_segments,
                              layer_sizes=[16, 16], activations=["relu", "relu"],
                              dropout=0.5, learning_rate=0.01,
                              epochs=200, patience=50):
    """
    Create and train a GCN model for hydrocarbon potential prediction
    
    Parameters:
    -----------
    node_data : pandas.DataFrame
        DataFrame containing node connections
    edge_data : pandas.DataFrame
        DataFrame containing edge data
    facies_segments : list
        List of lists containing indices of continuous segments
    layer_sizes : list
        List of hidden layer sizes
    activations : list
        List of activation functions for each layer
    dropout : float
        Dropout rate
    learning_rate : float
        Learning rate for Adam optimizer
    epochs : int
        Number of training epochs
    patience : int
        Patience for early stopping
        
    Returns:
    --------
    VHydroGCN
        Trained model
    pandas.DataFrame
        Training history
    pandas.DataFrame
        Predictions
    """
    # Create model
    model = VHydroGCN(node_data, edge_data, facies_segments)
    
    # Build model
    model.build_model(
        layer_sizes=layer_sizes,
        activations=activations,
        dropout=dropout,
        learning_rate=learning_rate
    )
    
    # Train model
    history = model.train(epochs=epochs, patience=patience)
    
    # Evaluate model
    model.evaluate()
    
    # Generate predictions
    predictions = model.predict()
    
    return model, history, predictions

def simulate_gcn_prediction(edge_data, accuracy=0.9):
    """
    Simulate GCN prediction results without training a model
    (for demonstration purposes)
    
    Parameters:
    -----------
    edge_data : pandas.DataFrame
        DataFrame containing edge data with 'DEPTH' and 'PET_label' columns
    accuracy : float
        Simulated accuracy level
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with true and predicted labels
    """
    # Create a copy of the data
    predicted_df = edge_data.copy()
    
    # Create a list of possible labels
    labels = ['Very_Low', 'Low', 'Moderate', 'High', 'Very_High']
    
    # Initialize predicted column
    predicted_df['Predicted_PET'] = ''
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # For each row, decide whether to keep the true label or assign a random one
    for i in range(len(predicted_df)):
        if np.random.random() > accuracy:
            # Assign a different random label
            current_label = predicted_df.loc[i, 'PET_label']
            other_labels = [l for l in labels if l != current_label]
            predicted_df.loc[i, 'Predicted_PET'] = np.random.choice(other_labels)
        else:
            # Keep the true label
            predicted_df.loc[i, 'Predicted_PET'] = predicted_df.loc[i, 'PET_label']
    
    return predicted_df