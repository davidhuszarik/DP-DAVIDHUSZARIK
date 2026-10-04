import streamlit as st


def render():
    st.header("3. Vylepšený BKW pomocou Krycích Kódov (Covering Codes)")
    st.latex(r"g_i' = c_i + e_i'")

    k_prime = st.number_input(
        "Dĺžka stĺpca po BKW (k'')", min_value=10, max_value=200, value=75
    )
    code_dim = st.number_input(
        "Dimenzia krycieho kódu (l)", min_value=5, max_value=50, value=25
    )
    avg_dist = k_prime * 0.25
    bias_code = 1 - (2 * avg_dist / k_prime)

    st.write(f"Priemerná vzdialenosť $d \\approx$ **{avg_dist:.1f}**")
    st.write(f"Dodatočný bias $\\epsilon' \\approx$ **{bias_code:.3f}**")