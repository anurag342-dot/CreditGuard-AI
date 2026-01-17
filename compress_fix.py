import joblib
import os

print("Loading file... this may take a second.")
# Load the big file
selector = joblib.load('feature_selector.pkl')

# Save it again with high compression
print("Compressing and saving...")
joblib.dump(selector, 'feature_selector.pkl', compress=9)

# Check new size
size_mb = os.path.getsize('feature_selector.pkl') / (1024 * 1024)
print(f"✅ Success! New file size: {size_mb:.2f} MB")