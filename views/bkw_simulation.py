import numpy as np
import plotly.express as px
import streamlit as st


@st.cache_data
def run_bkw_simulation(sim_k, sim_b, sim_n, sim_eta, bkw_mode="LF1"):
    x_true = np.random.randint(0, 2, size=sim_k)
    G_curr = np.random.randint(0, 2, size=(sim_k, sim_n))
    e_curr = (np.random.rand(sim_n) < sim_eta).astype(int)
    z_curr = (x_true @ G_curr + e_curr) % 2

    steps_history = []
    steps_history.append(
        {
            "step": 0,
            "G": G_curr,
            "z": z_curr,
            "e": e_curr,
            "bias": 1 - 2 * sim_eta,
            "eta": sim_eta,
            "operations": [],
            "partitions": {},
            "elim_rows": (0, 0),
        }
    )

    t_steps = sim_k // sim_b
    curr_bias = 1 - 2 * sim_eta

    for t in range(1, t_steps + 1):
        elim_start_row = sim_k - t * sim_b
        if elim_start_row < 0:
            break

        sub_matrix = G_curr[elim_start_row : elim_start_row + sim_b, :]
        col_tuples = [
            tuple(int(x) for x in sub_matrix[:, col_idx])
            for col_idx in range(sub_matrix.shape[1])
        ]

        partitions = {}
        for col_idx, block_val in enumerate(col_tuples):
            partitions.setdefault(block_val, []).append(col_idx)

        new_G_cols = []
        new_z_vals = []
        new_e_vals = []
        operations_log = []

        for block_val, col_indices in partitions.items():
            if len(col_indices) < 2:
                continue

            if bkw_mode == "LF1":
                base_idx = col_indices[0]
                target_indices = np.array(col_indices[1:])

                combined_cols = (
                    G_curr[:, target_indices] ^ G_curr[:, base_idx, None]
                )
                combined_z = z_curr[target_indices] ^ z_curr[base_idx]
                combined_e = e_curr[target_indices] ^ e_curr[base_idx]

                new_G_cols.append(combined_cols)
                new_z_vals.append(combined_z)
                new_e_vals.append(combined_e)

                for idx, target_idx in enumerate(target_indices):
                    operations_log.append(
                        {
                            "block": block_val,
                            "base_idx": base_idx,
                            "target_idx": target_idx,
                            "e_base": int(e_curr[base_idx]),
                            "e_target": int(e_curr[target_idx]),
                            "e_res": int(combined_e[idx]),
                        }
                    )
            elif bkw_mode == "LF2":
                n_elems = len(col_indices)
                for i in range(n_elems):
                    for j in range(i + 1, n_elems):
                        idx1 = col_indices[i]
                        idx2 = col_indices[j]

                        combined_col = (
                            G_curr[:, idx1] ^ G_curr[:, idx2]
                        )[:, None]
                        combined_z = np.array([z_curr[idx1] ^ z_curr[idx2]])
                        combined_e = np.array([e_curr[idx1] ^ e_curr[idx2]])

                        new_G_cols.append(combined_col)
                        new_z_vals.append(combined_z)
                        new_e_vals.append(combined_e)

                        operations_log.append(
                            {
                                "block": block_val,
                                "base_idx": idx1,
                                "target_idx": idx2,
                                "e_base": int(e_curr[idx1]),
                                "e_target": int(e_curr[idx2]),
                                "e_res": int(combined_e[0]),
                            }
                        )

        if not new_G_cols:
            break

        G_curr = np.hstack(new_G_cols)
        z_curr = np.concatenate(new_z_vals)
        e_curr = np.concatenate(new_e_vals)
        curr_bias = curr_bias**2
        curr_eta = 0.5 * (1 - curr_bias)

        steps_history.append(
            {
                "step": t,
                "G": G_curr,
                "z": z_curr,
                "e": e_curr,
                "bias": curr_bias,
                "eta": curr_eta,
                "operations": operations_log,
                "partitions": partitions,
                "elim_rows": (elim_start_row, elim_start_row + sim_b),
            }
        )

    return x_true, steps_history


def render():
    st.header("2. BKW Algoritmus: Simulácia a Sledovanie Operácií")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        sim_k = st.number_input(
            "Dimenzia kľúča (k)", min_value=4, max_value=16, value=8
        )
    with col2:
        sim_b = st.number_input(
            "Veľkosť bloku (b)", min_value=1, max_value=4, value=2
        )
    with col3:
        sim_n = st.number_input(
            "Počet vzoriek (n)", min_value=15, max_value=500, value=80
        )
    with col4:
        sim_eta = st.slider("Šum (η)", 0.01, 0.3, 0.08, step=0.01)
    with col5:
        bkw_mode = st.selectbox(
            "Spôsob spájania",
            ["LF1", "LF2"],
            help="LF1 sčítava 1 reprezentanta s ostatnými. LF2 sčítava všetky možné dvojice v partícii.",
        )

    x_true, hist = run_bkw_simulation(
        sim_k, sim_b, sim_n, sim_eta, bkw_mode=bkw_mode
    )
    st.success(f"Simulácia ({bkw_mode}) vypočítaná. Tajný kľúč x = {x_true}")

    if len(hist) > 1:
        st.divider()
        step_idx = st.slider(
            "Vyberte krok BKW na preskúmanie detailov:", 1, len(hist) - 1, 1
        )
        step_data = hist[step_idx]

        st.subheader(f"🔍 Detailný priebeh Operácií v Kroku {step_idx} ({bkw_mode})")
        r_start, r_end = step_data["elim_rows"]
        st.info(
            f"Eliminujeme riadky **{r_start} až {r_end - 1}** (veľkosť bloku b = {sim_b})."
        )

        st.markdown("#### 1. Rozdelenie do Partícií")
        part_summary = []
        for b_val, indices in step_data["partitions"].items():
            n_elems = len(indices)
            pairs_count = (
                n_elems - 1 if bkw_mode == "LF1" else (n_elems * (n_elems - 1)) // 2
            )
            part_summary.append(
                {
                    "Hodnota bloku b": str(b_val),
                    "Počet stĺpcov v skupine": n_elems,
                    "Vytvorené nové dvojice": max(0, pairs_count),
                }
            )
        st.table(part_summary)

        st.markdown("#### 2. Log XOR operácií")
        if step_data["operations"]:
            op_idx = st.number_input(
                "Vyberte index operácie:",
                min_value=0,
                max_value=len(step_data["operations"]) - 1,
                value=0,
            )
            op = step_data["operations"][op_idx]
            prev_G = hist[step_idx - 1]["G"]

            g_base = prev_G[:, op["base_idx"]]
            g_target = prev_G[:, op["target_idx"]]
            g_res = g_target ^ g_base

            col_op1, col_op2, col_op3 = st.columns(3)
            with col_op1:
                st.markdown(f"**Prvá vzorka $g_{{{op['base_idx']}}}$:**")
                st.code(str(g_base))
                st.caption(f"Šum $e$: {op['e_base']}")
            with col_op2:
                st.markdown(f"**Druhá vzorka $g_{{{op['target_idx']}}}$:**")
                st.code(str(g_target))
                st.caption(f"Šum $e$: {op['e_target']}")
            with col_op3:
                st.markdown("**Výsledok XOR ($g_{new}$):**")
                st.code(str(g_res))
                st.caption(f"Nový šum $e_{{new}}$: {op['e_res']}")

        st.markdown("#### 3. Zmena štruktúry matice $G$")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown(f"**Matica pred krokom {step_idx}:**")
            fig_before = px.imshow(
                hist[step_idx - 1]["G"],
                color_continuous_scale=["white", "navy"],
            )
            fig_before.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_before, use_container_width=True)

        with col_g2:
            st.markdown(f"**Matica po eliminácii (Krok {step_idx}):**")
            fig_after = px.imshow(
                step_data["G"], color_continuous_scale=["white", "navy"]
            )
            fig_after.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_after, use_container_width=True)
    else:
        st.warning(
            "Vzorky nestačili na vykonanie ani jedného kroku BKW. Zvýšte počet vzoriek (n)."
        )