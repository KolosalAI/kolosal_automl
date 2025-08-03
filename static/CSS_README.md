# CSS Styling System

This document explains how the CSS styling system works for the AutoML platform.

## File Structure

- **`static/styles.css`** - Main stylesheet containing all UI styles
- **`app.py`** - Main application file with `load_css_file()` function

## How It Works

1. **CSS Loading**: The `load_css_file()` function in `app.py` loads styles from `static/styles.css`
2. **Gradio Integration**: The loaded CSS is passed to the Gradio interface via the `css` parameter
3. **Fallback**: If the CSS file is missing, the app continues with minimal default styles

## CSS Classes

The stylesheet defines various UI component classes:

### Container Classes
- `.gradio-container` - Main application container
- `.inference-container` - Inference server section

### Card Components
- `.info-card` - Information display cards
- `.success-card` - Success message cards
- `.error-card` - Error message cards
- `.metric-box` - Metrics display boxes
- `.model-info-card` - Model information cards

### Navigation & Steps
- `.tab-nav button` - Tab navigation buttons
- `.step-indicator` - Step indicator badges
- `.workflow-step` - Workflow step containers

### Special Components
- `.quick-start` - Quick start guide section
- `.feature-grid` - Feature display grid
- `.feature-card` - Individual feature cards

### Server Status Components
- `.server-status` - Server status displays
- `.status-indicator` - Status indicator dots
- `.server-logs` - Server log displays
- `.control-button` - Action buttons

## Customization

To modify the styling:

1. Edit `static/styles.css` 
2. Restart the application to load changes
3. Changes will apply to all UI components using the defined classes

## Benefits

- **Separation of Concerns**: CSS is separate from Python code
- **Easy Maintenance**: All styles in one centralized file
- **Better Performance**: External CSS can be cached by browsers
- **Cleaner Code**: Python code focuses on functionality, not styling
