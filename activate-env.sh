# Detect if the script is being sourced
(return 0 2>/dev/null) && sourced=1 || sourced=0

if [ $sourced -eq 0 ]; then
    echo "Error: This script needs to be sourced. Run:"
    echo "    source activate-env.sh"
    echo "    or"
    echo "    . activate-env.sh"
    exit 1
fi

# Check if Miniconda is installed
if ! command -v conda &> /dev/null; then
    echo "Miniconda is not installed. Please install Miniconda and try again."
    exit 1
fi

# Check if environment.yml exists
if [ ! -f "environment.yml" ]; then
    echo "environment.yml not found in the current directory. Please provide an environment.yml file."
    exit 1
fi

# Extract environment name from environment.yml
env_name=$(grep "^name:" environment.yml | awk '{print $2}')

if [ -z "$env_name" ]; then
    echo "Environment name not found in environment.yml. Please ensure the file has a 'name' field."
    exit 1
fi

# Check if Conda is initialized
if ! conda info &> /dev/null; then
    echo "Conda is not initialized. Run 'conda init' and restart your shell."
    exit 1
fi

# Check if the environment already exists
if conda env list | grep -q "^$env_name\s"; then
    echo "Activating existing environment: $env_name"
else
    echo "Environment $env_name not found. Creating it from environment.yml..."
    # Initialize conda for the shell
    eval "$(conda shell.bash hook)"
    
    conda env create -f environment.yml

    if [ $? -ne 0 ]; then
        echo "Failed to create the environment. Check your environment.yml for errors."
        exit 1
    fi

    echo "Environment $env_name created successfully."
fi

# Initialize conda and activate environment
. $(conda info --base)/etc/profile.d/conda.sh
conda activate "$env_name"

# Confirm activation (but don't exit since we're sourcing)
if [ "$CONDA_DEFAULT_ENV" = "$env_name" ]; then
    echo "Environment $env_name is now active."
else
    echo "Failed to activate environment $env_name."
    return 1
fi
