python -m venv rcpsa_env
source rcpsa_env/bin/activate
brew install graphviz
export CFLAGS="-I$(brew --prefix graphviz)/include"
export LDFLAGS="-L$(brew --prefix graphviz)/lib"
pip install --upgrade pip
pip install -r requirements.txt