from core.sistema_transporte import SistemaTransporte

def construir_sistema():
    sistema = SistemaTransporte()

    # =========================
    # LINEA A (COMPLETA)
    # =========================
    linea_a = [
        "Niquia", "Bello", "Madera", "Acevedo", "Tricentenario",
        "Caribe", "Universidad", "Hospital", "Prado",
        "Parque Berrio", "San Antonio", "Alpujarra",
        "Exposiciones", "Industriales", "Poblado",
        "Aguacatala", "Ayura", "Envigado", "Itagui",
        "Sabaneta", "La Estrella"
    ]

    for i in range(len(linea_a) - 1):
        sistema.conectar(linea_a[i], linea_a[i+1], "Metro", 3, 0)

    # =========================
    # LINEA B (COMPLETA)
    # =========================
    linea_b = [
        "San Javier", "Santa Lucia", "Floresta",
        "Estadio", "Suramericana", "Cisneros", "San Antonio"
    ]

    for i in range(len(linea_b) - 1):
        sistema.conectar(linea_b[i], linea_b[i+1], "Metro", 3, 0)

    # =========================
    # CENTRO DE MEDELLÍN
    # =========================
    sistema.conectar("Prado", "Centro Medellin", "Caminando", 10, 0)
    sistema.conectar("Parque Berrio", "Centro Medellin", "Caminando", 3, 0)
    sistema.conectar("San Antonio", "Centro Medellin", "Caminando", 5, 0)
    sistema.conectar("Alpujarra", "Centro Medellin", "Caminando", 10, 0)

    # =========================
    # MUNICIPIOS VALLE DE ABURRÁ
    # =========================
    # Conexiones intermunicipales (Buses)
    sistema.conectar("Bello", "Copacabana", "Bus", 15, 2800)
    sistema.conectar("Copacabana", "Girardota", "Bus", 15, 2800)
    sistema.conectar("Girardota", "Barbosa", "Bus", 20, 2800)

    sistema.conectar("Sabaneta", "Caldas", "Bus", 20, 2800)

    # =========================
    # BUSES INTEGRADOS (EJEMPLOS REALES)
    # =========================
    sistema.conectar("Acevedo", "Santo Domingo", "Cable", 10, 0)
    sistema.conectar("San Javier", "La Aurora", "Cable", 10, 0)

    sistema.conectar("Caribe", "Terminal Norte", "Bus", 5, 2800)
    sistema.conectar("Poblado", "Terminal Sur", "Bus", 5, 2800)

    sistema.conectar("Universidad", "UNAL", "Bus", 5, 2800)
    sistema.conectar("Industriales", "UPB", "Bus", 7, 2800)

    # =========================
    # TAXIS (CONEXIONES RÁPIDAS)
    # =========================
    sistema.conectar("Centro Medellin", "Poblado", "Taxi", 15, 10000)
    sistema.conectar("Bello", "Poblado", "Taxi", 25, 15000)
    sistema.conectar("Envigado", "Centro Medellin", "Taxi", 15, 10000)

    # =========================
    # LUGARES IMPORTANTES
    # =========================
    sistema.conectar("Estadio", "Atanasio Girardot", "Bus", 5, 2800)
    sistema.conectar("Universidad", "Jardin Botanico", "Bus", 5, 2800)
    sistema.conectar("Hospital", "Parque Explora", "Bus", 5, 2800)
    sistema.conectar("Exposiciones", "Plaza Mayor", "Bus", 5, 2800)
    sistema.conectar("Poblado", "Parque Lleras", "Bus", 5, 2800)

    return sistema