"""Corre toda la tarea en orden: tablas en tablas/ y figuras en figuras/."""
import parte0_simulacion
import parte1_limpieza
import parte2_calendario
import parte3_descomp
import parte4_sarima
import parte5_pronostico
import parte6_rfm

if __name__ == "__main__":
    for parte in (parte0_simulacion, parte1_limpieza, parte2_calendario, parte3_descomp,
                  parte4_sarima, parte5_pronostico, parte6_rfm):
        print(f"\n===== {parte.__name__} =====")
        parte.main()
