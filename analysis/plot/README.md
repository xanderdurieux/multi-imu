## master-thesis/analysis/plot

Plots accelerometer, gyroscope, and magnetometer values from processed CSV data.

### Preferred usage (as a module)

From the `analysis/` folder:

```bash
uv run -m plot.plot_session data/processed/<session_name>/<filename>.csv
# or
python3 -m plot.plot_session data/processed/<session_name>/<filename>.csv
```