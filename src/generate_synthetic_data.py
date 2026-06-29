import numpy as np
import pandas as pd


def generate_synthetic_data(num_records=1000, output_filename="synthetic_data.csv"):
    # Fix the random seed for reproducible results
    np.random.seed(42)

    # 1. Chelating agent (C): float [0-10], 2 decimal places
    C = np.round(np.random.uniform(0.0, 10.0, num_records), 2)

    # 2. Catalyst mix (D): derived from C + D = 10 -> D = 10 - C
    D = np.round(10.0 - C, 2)

    # 3. Ratio (C/D): handles edge case division by zero if D is 0.0
    # Uses np.where to replace division by zero with NaN or inf safely
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.round(C / D, 2)
        # Optional: Replace infinity values if D is exactly 0.0
        ratio = np.where(np.isinf(ratio), np.nan, ratio)

    # 4. Calcination temperature: integer [150-800]
    calcination_temp = np.random.randint(150, 801, size=num_records)

    # 5. Fe (A): integer [0-100]
    A = np.random.randint(0, 101, size=num_records)

    # 6. W (B): derived from A + B = 100 -> B = 100 - A
    B = 100 - A

    # Assemble into a structured DataFrame
    df = pd.DataFrame(
        {
            "Chelating agent (C)": C,
            "Catalyst mix (D)": D,
            "Chelating catalyst mix ratio (C/D)": ratio,
            "Calcination temperature (oC)": calcination_temp,
            "Fe (A)": A,
            "W (B) wt%": B,
        }
    )

    # Save to a local CSV file without row index indices
    df.to_csv(output_filename, index=False)
    print(f"Successfully generated and saved {num_records} records to '{output_filename}'.")
    return df


# Generate 120 records as an example
df_synthetic = generate_synthetic_data(num_records=120)
print(df_synthetic.head())
