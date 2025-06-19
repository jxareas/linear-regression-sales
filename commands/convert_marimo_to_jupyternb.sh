cd ../ || { echo "Failed to cd ../"; exit 1; }

FILE=./notebooks/M2_AG_Grupo12.ipynb

if [ -f "$FILE" ]; then
  echo "Removing $FILE..."
  if rm "$FILE"; then
    echo "Successfully removed $FILE."
  else
    echo "Failed to remove $FILE."
  fi
else
  echo "File $FILE not found, skipping removal."
fi

echo "Running marimo export command..."
if marimo export ipynb ./scripts/M2_AG_Grupo12.py -o ./notebooks/M2_AG_Grupo12.ipynb; then
  echo "marimo export completed successfully."
else
  echo "marimo export command failed."
fi
