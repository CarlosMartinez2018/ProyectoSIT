import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from langchain.agents import create_agent
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import json
from pathlib import Path
import glob
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

class AmadeusFlightAnalytics:
    """Módulo de análisis estadístico para datos Amadeus de Medellín"""
    
    def __init__(self, searches_df=None, bookings_df=None):
        self.searches_df = searches_df
        self.bookings_df = bookings_df
        
        # Procesar datos al inicializar
        if self.searches_df is not None:
            self._process_searches()
        if self.bookings_df is not None:
            self._process_bookings()
    
    def _process_searches(self):
        """Preprocesa datos de búsquedas"""
        self.searches_df['creation_date'] = pd.to_datetime(self.searches_df['creation_date'])
        self.searches_df['ond_orig_dep_date'] = pd.to_datetime(self.searches_df['ond_orig_dep_date'])
        self.searches_df['search_month'] = self.searches_df['creation_date'].dt.month
        self.searches_df['search_year'] = self.searches_df['creation_date'].dt.year
        self.searches_df['dep_month'] = self.searches_df['ond_orig_dep_date'].dt.month
        
        # Calcular lead_time si no existe
        if 'lead_time' not in self.searches_df.columns:
            self.searches_df['lead_time'] = (self.searches_df['ond_orig_dep_date'] - self.searches_df['creation_date']).dt.days
    
    def _process_bookings(self):
        """Preprocesa datos de reservas"""
        self.bookings_df['creation_date'] = pd.to_datetime(self.bookings_df['creation_date'])
        self.bookings_df['trip_dep_date'] = pd.to_datetime(self.bookings_df['trip_dep_date'])
        self.bookings_df['booking_month'] = self.bookings_df['creation_date'].dt.month
        self.bookings_df['booking_year'] = self.bookings_df['creation_date'].dt.year
    
    def analisis_demografico_origen(self, query=""):
        """Análisis demográfico por origen geográfico y agencias"""
        analisis = {
            "resumen_ejecutivo": {},
            "mercados_origen": {},
            "perfil_agencias": {},
            "segmentacion": {}
        }
        
        if self.bookings_df is None:
            return "No hay datos de bookings disponibles"
        
        # Análisis de mercados origen
        top_paises = self.bookings_df['trip_board_ctry_name'].value_counts().head(10)
        top_ciudades = self.bookings_df['trip_board_city_code'].value_counts().head(10)
        
        analisis["mercados_origen"] = {
            "top_10_paises": top_paises.to_dict(),
            "top_10_ciudades": top_ciudades.to_dict(),
            "total_paises_unicos": int(self.bookings_df['trip_board_ctry_code'].nunique()),
            "total_ciudades_unicas": int(self.bookings_df['trip_board_city_code'].nunique())
        }
        
        # Perfil de agencias de viaje
        if 'travel_agency_profile' in self.bookings_df.columns:
            agency_dist = self.bookings_df['travel_agency_profile'].value_counts()
            analisis["perfil_agencias"] = {
                "distribucion": agency_dist.to_dict(),
                "agencia_dominante": agency_dist.idxmax()
            }
        
        # Clasificación Business vs Leisure
        if 'business_leisure' in self.bookings_df.columns:
            biz_leisure = self.bookings_df['business_leisure'].value_counts()
            total = biz_leisure.sum()
            analisis["segmentacion"]["business_leisure"] = {
                "distribucion": biz_leisure.to_dict(),
                "porcentajes": {k: f"{(v/total*100):.1f}%" for k, v in biz_leisure.items()}
            }
        
        # Tipo de viaje (One-way, Return, etc)
        if 'trip_class' in self.bookings_df.columns:
            trip_type = self.bookings_df['trip_class'].value_counts()
            analisis["segmentacion"]["tipo_viaje"] = trip_type.to_dict()
        
        # Tamaño de grupos
        if 'nb_pax_together' in self.bookings_df.columns:
            analisis["segmentacion"]["tamano_grupo"] = {
                "promedio_pasajeros": float(self.bookings_df['nb_pax_together'].mean()),
                "viajeros_solos": int((self.bookings_df['nb_pax_together'] == 1).sum()),
                "grupos_2_personas": int((self.bookings_df['nb_pax_together'] == 2).sum()),
                "grupos_3plus": int((self.bookings_df['nb_pax_together'] >= 3).sum())
            }
        
        # Resumen ejecutivo
        analisis["resumen_ejecutivo"] = {
            "total_reservas": len(self.bookings_df),
            "total_pasajeros": int(self.bookings_df['ond_pax'].sum()) if 'ond_pax' in self.bookings_df.columns else 0,
            "mercado_principal": top_paises.idxmax() if len(top_paises) > 0 else "N/A",
            "ciudad_principal": top_ciudades.idxmax() if len(top_ciudades) > 0 else "N/A"
        }
        
        return json.dumps(analisis, indent=2, ensure_ascii=False)
    
    def analisis_temporal_demanda(self, query=""):
        """Análisis temporal: estacionalidad, lead time, anticipación"""
        analisis = {
            "estacionalidad_busquedas": {},
            "estacionalidad_reservas": {},
            "anticipacion_compra": {},
            "duracion_estancia": {}
        }
        
        # Análisis de búsquedas
        if self.searches_df is not None:
            # Meses más buscados para viajar
            dep_months = self.searches_df['dep_month'].value_counts().sort_index()
            analisis["estacionalidad_busquedas"] = {
                "viajes_mas_buscados_por_mes": dep_months.to_dict(),
                "temporada_alta": [int(m) for m in dep_months.nlargest(3).index],
                "temporada_baja": [int(m) for m in dep_months.nsmallest(3).index]
            }
            
            # Lead time (anticipación de búsqueda)
            if 'lead_time' in self.searches_df.columns:
                analisis["anticipacion_compra"]["lead_time_busquedas"] = {
                    "promedio_dias": float(self.searches_df['lead_time'].mean()),
                    "mediana_dias": float(self.searches_df['lead_time'].median()),
                    "anticipacion_corta_0_7dias": int((self.searches_df['lead_time'] <= 7).sum()),
                    "anticipacion_media_8_30dias": int(((self.searches_df['lead_time'] > 7) & (self.searches_df['lead_time'] <= 30)).sum()),
                    "anticipacion_larga_30plus": int((self.searches_df['lead_time'] > 30).sum())
                }
            
            # Duración de estancia
            if 'stay_duration' in self.searches_df.columns:
                # Filtrar one-way (-1)
                stays = self.searches_df[self.searches_df['stay_duration'] >= 0]['stay_duration']
                analisis["duracion_estancia"]["busquedas"] = {
                    "promedio_noches": float(stays.mean()) if len(stays) > 0 else 0,
                    "estancias_cortas_1_3noches": int((stays <= 3).sum()),
                    "estancias_medias_4_7noches": int(((stays > 3) & (stays <= 7)).sum()),
                    "estancias_largas_7plus": int((stays > 7).sum())
                }
        
        # Análisis de reservas
        if self.bookings_df is not None:
            # Meses de viaje reservados
            trip_months = self.bookings_df.groupby(self.bookings_df['trip_dep_date'].dt.month).size()
            analisis["estacionalidad_reservas"] = {
                "reservas_por_mes_viaje": trip_months.to_dict(),
                "meses_pico_reservas": [int(m) for m in trip_months.nlargest(3).index]
            }
            
            # Lead time de reservas
            self.bookings_df['booking_lead_time'] = (
                self.bookings_df['trip_dep_date'] - self.bookings_df['creation_date']
            ).dt.days
            
            analisis["anticipacion_compra"]["lead_time_reservas"] = {
                "promedio_dias": float(self.bookings_df['booking_lead_time'].mean()),
                "mediana_dias": float(self.bookings_df['booking_lead_time'].median())
            }
            
            # Duración de estancia (days_at_destination)
            if 'days_at_destination' in self.bookings_df.columns:
                stays_book = self.bookings_df[self.bookings_df['days_at_destination'] > 0]['days_at_destination']
                analisis["duracion_estancia"]["reservas"] = {
                    "promedio_noches": float(stays_book.mean()) if len(stays_book) > 0 else 0,
                    "mediana_noches": float(stays_book.median()) if len(stays_book) > 0 else 0
                }
        
        return json.dumps(analisis, indent=2, ensure_ascii=False)
    
    def analisis_economico_precios(self, query=""):
        """Análisis económico: precios, revenue, cabinas"""
        analisis = {
            "estructura_precios": {},
            "revenue": {},
            "preferencias_cabina": {},
            "valor_por_segmento": {}
        }
        
        if self.bookings_df is None:
            return "No hay datos de bookings disponibles"
        
        # Análisis de precios
        if 'avg_indicative_price' in self.bookings_df.columns:
            prices = self.bookings_df[self.bookings_df['avg_indicative_price'] > 0]['avg_indicative_price']
            
            analisis["estructura_precios"] = {
                "precio_promedio_usd": float(prices.mean()),
                "precio_mediana_usd": float(prices.median()),
                "precio_min_usd": float(prices.min()),
                "precio_max_usd": float(prices.max()),
                "desviacion_estandar": float(prices.std()),
                "percentil_25": float(prices.quantile(0.25)),
                "percentil_75": float(prices.quantile(0.75))
            }
            
            # Revenue total
            total_pax = self.bookings_df['ond_pax'].sum() if 'ond_pax' in self.bookings_df.columns else len(self.bookings_df)
            total_revenue = (self.bookings_df['avg_indicative_price'] * self.bookings_df['ond_pax']).sum() if 'ond_pax' in self.bookings_df.columns else prices.sum()
            
            analisis["revenue"] = {
                "revenue_total_usd": float(total_revenue),
                "revenue_promedio_por_reserva": float(total_revenue / len(self.bookings_df))
            }
            
            # Revenue por país origen
            if 'trip_board_ctry_name' in self.bookings_df.columns:
                rev_by_country = self.bookings_df.groupby('trip_board_ctry_name')['avg_indicative_price'].agg(['sum', 'mean'])
                top_revenue_countries = rev_by_country.nlargest(5, 'sum')
                analisis["valor_por_segmento"]["top_paises_revenue"] = {
                    country: {
                        "revenue_total": float(row['sum']),
                        "precio_promedio": float(row['mean'])
                    }
                    for country, row in top_revenue_countries.iterrows()
                }
        
        # Análisis de cabinas
        if 'ond_cab_class' in self.bookings_df.columns:
            cabin_dist = self.bookings_df['ond_cab_class'].value_counts()
            cabin_names = {'Y': 'Economy', 'W': 'Premium Economy', 'C': 'Business', 'F': 'First'}
            
            analisis["preferencias_cabina"] = {
                "distribucion": {cabin_names.get(k, k): int(v) for k, v in cabin_dist.items()},
                "porcentaje_premium": f"{((cabin_dist.get('C', 0) + cabin_dist.get('F', 0)) / cabin_dist.sum() * 100):.1f}%"
            }
            
            # Precio promedio por cabina
            if 'avg_indicative_price' in self.bookings_df.columns:
                price_by_cabin = self.bookings_df.groupby('ond_cab_class')['avg_indicative_price'].mean()
                analisis["preferencias_cabina"]["precio_promedio_por_cabina"] = {
                    cabin_names.get(k, k): float(v) for k, v in price_by_cabin.items()
                }
        
        return json.dumps(analisis, indent=2, ensure_ascii=False)
    
    def analisis_conversion_funnel(self, query=""):
        """Análisis de conversión: búsquedas a reservas"""
        analisis = {
            "metricas_conversion": {},
            "analisis_por_origen": {},
            "oportunidades": []
        }
        
        if self.searches_df is None or self.bookings_df is None:
            return "Se requieren ambos datasets (searches y bookings) para análisis de conversión"
        
        # Métricas generales
        # Métricas generales
        total_searches = self.searches_df['nb_pax_together'].sum() if 'nb_pax_together' in self.searches_df.columns else len(self.searches_df)
        total_bookings = self.bookings_df['ond_pax'].sum() if 'ond_pax' in self.bookings_df.columns else len(self.bookings_df)
        total_bookings = self.bookings_df['ond_pax'].sum() if 'ond_pax' in self.bookings_df.columns else len(self.bookings_df)
        
        conversion_rate = (total_bookings / total_searches * 100) if total_searches > 0 else 0
        
        analisis["metricas_conversion"] = {
            "total_pasajeros_buscados": int(total_searches),
            "total_pasajeros_reservados": int(total_bookings),
            "tasa_conversion": f"{conversion_rate:.2f}%",
            "busquedas_sin_convertir": int(total_searches - total_bookings)
        }
        
        # Conversión por origen
        if 'ond_orig_ctry_code' in self.searches_df.columns and 'trip_board_ctry_code' in self.bookings_df.columns:
            search_by_country = self.searches_df.groupby('ond_orig_ctry_code')['nb_pax_together'].sum()
            booking_by_country = self.bookings_df.groupby('trip_board_ctry_code')['ond_pax'].sum()
            booking_by_country = self.bookings_df.groupby('trip_board_ctry_code')['ond_pax'].sum()
            
            # Top 5 países
            top_countries = search_by_country.nlargest(5).index
            for country in top_countries:
                searches = search_by_country.get(country, 0)
                bookings = booking_by_country.get(country, 0)
                conv = (bookings / searches * 100) if searches > 0 else 0
                
                analisis["analisis_por_origen"][country] = {
                    "busquedas": int(searches),
                    "reservas": int(bookings),
                    "conversion": f"{conv:.2f}%"
                }
        
        # Identificar oportunidades
        if conversion_rate < 30:
            analisis["oportunidades"].append({
                "tipo": "Conversión general baja",
                "descripcion": f"Solo {conversion_rate:.1f}% de las búsquedas se convierten en reservas",
                "recomendacion": "Implementar estrategias de retargeting y simplificar proceso de reserva"
            })
        
        if conversion_rate < 50:
            analisis["oportunidades"].append({
                "tipo": "Gap de conversión significativo",
                "descripcion": f"{100-conversion_rate:.1f}% de interesados no completan la reserva",
                "recomendacion": "Analizar barreras: precio, disponibilidad, UX del sitio"
            })
        
        return json.dumps(analisis, indent=2, ensure_ascii=False)
    
    def recomendar_publico_objetivo(self, query=""):
        """Recomienda segmentos de público objetivo para campañas"""
        recomendaciones = {
            "segmentos_prioritarios": [],
            "insights_clave": [],
            "quick_wins": []
        }
        
        if self.bookings_df is None:
            return "No hay datos de bookings disponibles"
        
        # Segmento 1: Mercados origen principales
        if 'trip_board_ctry_name' in self.bookings_df.columns:
            top_countries = self.bookings_df['trip_board_ctry_name'].value_counts().head(3)
            for i, (country, count) in enumerate(top_countries.items(), 1):
                pct = (count / len(self.bookings_df)) * 100
                recomendaciones["segmentos_prioritarios"].append({
                    "prioridad": i,
                    "segmento": f"Viajeros desde {country}",
                    "volumen": int(count),
                    "penetracion": f"{pct:.1f}%",
                    "justificacion": f"Mercado establecido con {pct:.1f}% del total de reservas"
                })
        
        # Segmento 2: Business vs Leisure
        if 'business_leisure' in self.bookings_df.columns:
            biz_leisure = self.bookings_df['business_leisure'].value_counts()
            for segment, count in biz_leisure.items():
                pct = (count / len(self.bookings_df)) * 100
                if pct > 20:  # Solo segmentos significativos
                    recomendaciones["segmentos_prioritarios"].append({
                        "segmento": f"Viajeros de {segment}",
                        "volumen": int(count),
                        "penetracion": f"{pct:.1f}%",
                        "justificacion": f"Segmento importante con {pct:.1f}% del mercado"
                    })
        
        # Segmento 3: Familias vs Individuales
        if 'nb_pax_together' in self.bookings_df.columns:
            avg_pax = self.bookings_df['nb_pax_together'].mean()
            grupos = (self.bookings_df['nb_pax_together'] >= 3).sum()
            pct_grupos = (grupos / len(self.bookings_df)) * 100
            
            if pct_grupos > 20:
                recomendaciones["segmentos_prioritarios"].append({
                    "segmento": "Familias y grupos (3+ personas)",
                    "volumen": int(grupos),
                    "penetracion": f"{pct_grupos:.1f}%",
                    "justificacion": "Oportunidad para paquetes familiares y promociones grupales"
                })
        
        # Insights basados en precios
        if 'avg_indicative_price' in self.bookings_df.columns:
            avg_price = self.bookings_df['avg_indicative_price'].mean()
            if avg_price < 400:
                recomendaciones["insights_clave"].append({
                    "insight": "Mercado sensible al precio",
                    "dato": f"Precio promedio: ${avg_price:.0f} USD",
                    "accion": "Enfatizar ofertas, descuentos y mejor relación calidad-precio en campañas"
                })
            else:
                recomendaciones["insights_clave"].append({
                    "insight": "Mercado premium",
                    "dato": f"Precio promedio: ${avg_price:.0f} USD",
                    "accion": "Destacar experiencias exclusivas, comodidad y servicios VIP"
                })
        
        # Quick wins basados en conversión
        if self.searches_df is not None:
            total_searches = len(self.searches_df)
            total_bookings = len(self.bookings_df)
            conv_rate = (total_bookings / total_searches * 100) if total_searches > 0 else 0
            
            if conv_rate < 40:
                gap_usuarios = int(total_searches * 0.3 - total_bookings)
                recomendaciones["quick_wins"].append({
                    "oportunidad": "Recuperar búsquedas perdidas",
                    "potencial": f"~{gap_usuarios:,} reservas adicionales",
                    "tactica": "Campañas de retargeting con ofertas limitadas en el tiempo"
                })
        
        return json.dumps(recomendaciones, indent=2, ensure_ascii=False)
    
    def estrategia_campanas_turismo(self, query=""):
        """Genera estrategia completa para campañas de turismo en Medellín"""
        estrategia = {
            "timing_optimo": {},
            "canales_recomendados": [],
            "mensajes_por_segmento": {},
            "tacticas_especificas": [],
            "kpis_sugeridos": {}
        }
        
        # 1. Timing óptimo
        if self.searches_df is not None and self.bookings_df is not None:
            # Meses de alta intención (búsquedas)
            search_months = self.searches_df['dep_month'].value_counts().nlargest(3)
            booking_months = self.bookings_df.groupby(self.bookings_df['trip_dep_date'].dt.month).size().nlargest(3)
            
            month_names = {1:'Enero', 2:'Febrero', 3:'Marzo', 4:'Abril', 5:'Mayo', 6:'Junio',
                          7:'Julio', 8:'Agosto', 9:'Septiembre', 10:'Octubre', 11:'Noviembre', 12:'Diciembre'}
            
            lead_time_val = self.searches_df['lead_time'].mean() if 'lead_time' in self.searches_df.columns else 0
            
            estrategia["timing_optimo"] = {
                "temporada_alta_busquedas": [month_names[m] for m in search_months.index],
                "temporada_alta_reservas": [month_names[m] for m in booking_months.index],
                "ventana_anticipacion": f"{lead_time_val:.0f} días promedio"
            }
        
        # 2. Canales recomendados basados en perfil
        if self.bookings_df is not None:
            if 'online_offline' in self.bookings_df.columns:
                online_pct = (self.bookings_df['online_offline'] == 'online').sum() / len(self.bookings_df) * 100
                
                if online_pct > 60:
                    estrategia["canales_recomendados"] = [
                        {"canal": "Google Ads Search", "prioridad": "Alta", "razon": "Alto volumen de búsquedas online"},
                        {"canal": "Meta Ads (Facebook/Instagram)", "prioridad": "Alta", "razon": "Segmentación precisa por demografía"},
                        {"canal": "YouTube Ads", "prioridad": "Media", "razon": "Contenido visual de Medellín"},
                        {"canal": "TikTok", "prioridad": "Media", "razon": "Tendencias y viralización"},
                        {"canal": "Email Marketing", "prioridad": "Media", "razon": "Retargeting de búsquedas"}
                    ]
                else:
                    estrategia["canales_recomendados"] = [
                        {"canal": "Agencias de viaje", "prioridad": "Alta", "razon": "Fuerte presencia offline"},
                        {"canal": "Google Ads", "prioridad": "Alta", "razon": "Capturar intención de búsqueda"},
                        {"canal": "Ferias de turismo", "prioridad": "Media", "razon": "Contacto directo B2B"}
                    ]
        
        # 3. Mensajes por segmento
        if 'business_leisure' in self.bookings_df.columns:
            biz_leisure = self.bookings_df['business_leisure'].value_counts()
            total = biz_leisure.sum()
            
            if 'business' in biz_leisure.index and (biz_leisure.get('business', 0) / total) > 0.3:
                estrategia["mensajes_por_segmento"]["Business"] = {
                    "mensaje_principal": "Medellín: Hub de negocios e innovación de Latinoamérica",
                    "puntos_clave": [
                        "Conectividad aérea internacional",
                        "Infraestructura hotelera 5 estrellas",
                        "Centros de convenciones modernos",
                        "Networking en ciudad innovadora"
                    ]
                }
            
            if 'leisure' in biz_leisure.index and (biz_leisure.get('leisure', 0) / total) > 0.5:
                estrategia["mensajes_por_segmento"]["Leisure"] = {
                    "mensaje_principal": "Medellín: Ciudad de eterna primavera y experiencias únicas",
                    "puntos_clave": [
                        "Clima perfecto todo el año",
                        "Gastronomía paisa auténtica",
                        "Cultura cafetera y paisajes",
                        "Vida nocturna y entretenimiento"
                    ]
                }
        
        # 4. Tácticas específicas
        if self.searches_df is not None and self.bookings_df is not None:
            total_searches = len(self.searches_df)
            total_bookings = len(self.bookings_df)
            conv_rate = (total_bookings / total_searches * 100) if total_searches > 0 else 0
            
            if conv_rate < 35:
                estrategia["tacticas_especificas"].append({
                    "tactica": "Campaña de retargeting agresiva",
                    "objetivo": f"Recuperar {100-conv_rate:.0f}% de búsquedas perdidas",
                    "acciones": [
                        "Pixel de seguimiento en búsquedas",
                        "Ofertas flash 24-48hrs post-búsqueda",
                        "Descuentos progresivos por tiempo limitado"
                    ],
                    "presupuesto_sugerido": "30-40% del budget digital"
                })
            
            # Lead time promedio
            if 'lead_time' in self.searches_df.columns:
                avg_lead = self.searches_df['lead_time'].mean()
                estrategia["tacticas_especificas"].append({
                    "tactica": "Calendario de campañas basado en anticipación",
                    "objetivo": f"Activar campaña {int(avg_lead)+7} días antes de fechas pico",
                    "acciones": [
                        f"Iniciar campañas {int(avg_lead)+14} días antes de temporada alta",
                        "Early bird discounts para reservas anticipadas",
                        "Urgencia: 'Solo X asientos a este precio'"
                    ]
                })
        
        # 5. KPIs sugeridos
        estrategia["kpis_sugeridos"] = {
            "alcance": [
                "Impresiones en mercados objetivo",
                "Alcance único por país origen"
            ],
            "consideracion": [
                "CTR (Click-Through Rate) en ads",
                "Tiempo en sitio de aterrizaje",
                "Páginas vistas por sesión"
            ],
            "conversion": [
                f"Tasa de conversión búsqueda-reserva (benchmark actual: {conv_rate:.1f}%)" if self.searches_df is not None and self.bookings_df is not None else "Tasa de conversión",
                "Costo por adquisición (CPA)",
                "Revenue por usuario (RPU)"
            ],
            "retencion": [
                "Retargeting conversion rate",
                "Email open rate",
                "Repeat booking rate"
            ]
        }
        
        return json.dumps(estrategia, indent=2, ensure_ascii=False)


class AmadeusDataLoader:
    """Cargador de datos Amadeus Air Market Data"""
    
    def __init__(self, searches_path=None, bookings_path=None):
        self.searches_df = None
        self.bookings_df = None
        self.analytics = None
        self.vectorstore = None
        
        if isinstance(searches_path, pd.DataFrame):
            self.searches_df = searches_path
            print(f"[OK] Air Searches cargadas desde DataFrame: {len(self.searches_df):,} registros")
        elif searches_path:
            self.load_searches(searches_path)
            
        if isinstance(bookings_path, pd.DataFrame):
            self.bookings_df = bookings_path
            print(f"[OK] Air Bookings cargadas desde DataFrame: {len(self.bookings_df):,} registros")
        elif bookings_path:
            self.load_bookings(bookings_path)
        
        # Inicializar analytics
        self.analytics = AmadeusFlightAnalytics(self.searches_df, self.bookings_df)
    
    def load_searches(self, filepath):
        """Carga Air Searches data desde Parquet"""
        try:
            self.searches_df = pd.read_parquet(filepath)
            print(f"[OK] Air Searches cargadas: {len(self.searches_df):,} registros")
            print(f"  Columnas: {list(self.searches_df.columns)}")
            if 'creation_date' in self.searches_df.columns:
                print(f"  Periodo: {self.searches_df['creation_date'].min()} a {self.searches_df['creation_date'].max()}")
            return self.searches_df
        except Exception as e:
            print(f" Error cargando Air Searches: {e}")
            return None
    
    def load_bookings(self, filepath):
        """Carga Air Bookings data desde Parquet"""
        try:
            self.bookings_df = pd.read_parquet(filepath)
            print(f"[OK] Air Bookings cargadas: {len(self.bookings_df):,} registros")
            print(f"  Columnas: {list(self.bookings_df.columns)}")
            if 'creation_date' in self.bookings_df.columns:
                print(f"  Periodo: {self.bookings_df['creation_date'].min()} a {self.bookings_df['creation_date'].max()}")
            return self.bookings_df
        except Exception as e:
            print(f" Error cargando Air Bookings: {e}")
            return None
    
    def create_vector_store(self):
        """Crea vectorstore para búsqueda semántica en los datos"""
        from langchain_core.documents import Document
        
        documents = []
        
        # Documentos de búsquedas (sample para no saturar)
        if self.searches_df is not None:
            sample_searches = self.searches_df.sample(min(1000, len(self.searches_df)))
            for _, row in sample_searches.iterrows():
                content = f"""Búsqueda de vuelo: 
                Origen: {row.get('ond_orig_city_code', 'N/A')} ({row.get('ond_orig_ctry_code', 'N/A')})
                Destino: Medellín (MDE)
                Fecha búsqueda: {row.get('creation_date', 'N/A')}
                Fecha viaje: {row.get('ond_orig_dep_date', 'N/A')}
                Pasajeros: {row.get('nb_pax_together', 'N/A')}
                Anticipación: {row.get('lead_time', 'N/A')} días"""
                
                
                # Convertir metadata a tipos simples (str, int, float, bool)
                meta = {"tipo": "busqueda"}
                for k, v in row.to_dict().items():
                    if isinstance(v, (pd.Timestamp, datetime)):
                        meta[k] = v.strftime('%Y-%m-%d')
                    elif v is None:
                        meta[k] = ""
                    else:
                        meta[k] = v

                doc = Document(
                    page_content=content,
                    metadata=meta
                )
                documents.append(doc)
        
        # Documentos de reservas (sample)
        if self.bookings_df is not None:
            sample_bookings = self.bookings_df.sample(min(1000, len(self.bookings_df)))
            for _, row in sample_bookings.iterrows():
                content = f"""Reserva de vuelo:
                Origen: {row.get('trip_board_city_code', 'N/A')} ({row.get('trip_board_ctry_name', 'N/A')})
                Destino: Medellín
                Fecha reserva: {row.get('creation_date', 'N/A')}
                Fecha viaje: {row.get('trip_dep_date', 'N/A')}
                Tipo: {row.get('business_leisure', 'N/A')}
                Precio: ${row.get('avg_indicative_price', 0)} USD
                Pasajeros: {row.get('nb_pax_together', 'N/A')}"""
                
                # Convertir metadata a tipos simples (str, int, float, bool)
                meta = {"tipo": "reserva"}
                for k, v in row.to_dict().items():
                    if isinstance(v, (pd.Timestamp, datetime)):
                        meta[k] = v.strftime('%Y-%m-%d')
                    elif v is None:
                        meta[k] = ""
                    else:
                        meta[k] = v
                        
                doc = Document(
                    page_content=content,
                    metadata=meta
                )
                documents.append(doc)
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=OpenAIEmbeddings()
        )
        
        print(f"[OK] Vector store creado con {len(documents):,} documentos")
        return self.vectorstore


class MedellinTourismAgent:
    """Agente de análisis turístico para Medellín - Datos Amadeus"""
    
    def __init__(self, api_key, data_loader, model="gpt-4"):
        self.data = data_loader
        self.model = model
        self.api_key = api_key
        self.chat_history = []

        
        # Definir las herramientas como funciones
        def analisis_origen_mercados(query: str = "") -> str:
            """Analiza mercados origen: países, ciudades, perfil de agencias (business/leisure, online/offline). Útil para identificar de dónde vienen los viajeros."""
            return self.data.analytics.analisis_demografico_origen(query)
        
        def analisis_temporal_estacionalidad(query: str = "") -> str:
            """Analiza patrones temporales: estacionalidad, lead time, anticipación de compra, duración de estancias. Útil para timing de campañas."""
            return self.data.analytics.analisis_temporal_demanda(query)
        
        def analisis_precios_revenue(query: str = "") -> str:
            """Analiza estructura de precios, revenue, preferencias de cabina. Útil para segmentación económica y pricing."""
            return self.data.analytics.analisis_economico_precios(query)
        
        def analisis_conversion(query: str = "") -> str:
            """Analiza conversión de búsquedas a reservas. Identifica gaps y oportunidades. Requiere ambos datasets."""
            return self.data.analytics.analisis_conversion_funnel(query)
        
        def recomendar_publico_objetivo(query: str = "") -> str:
            """Genera recomendaciones específicas de segmentos de público objetivo prioritarios para campañas basadas en datos reales."""
            return self.data.analytics.recomendar_publico_objetivo(query)
        
        def estrategia_campanas(query: str = "") -> str:
            """Genera estrategia completa de marketing: timing, canales, mensajes, tácticas y KPIs para mejorar turismo en Medellín."""
            return self.data.analytics.estrategia_campanas_turismo(query)
        
        # Lista de herramientas
        self.tools = [
            analisis_origen_mercados,
            analisis_temporal_estacionalidad,
            analisis_precios_revenue,
            analisis_conversion,
            recomendar_publico_objetivo,
            estrategia_campanas
        ]
        
        # System prompt
        self.system_prompt = """Eres un analista experto en turismo y estrategia de marketing para la ciudad de Medellín, Colombia.

Trabajas con datos reales de Amadeus Air Market Data que incluyen:
- Air Searches: Búsquedas de vuelos hacia Medellín (intención de viaje)
- Air Bookings: Reservas confirmadas hacia Medellín (conversión real)

Tu misión:
1. Generar análisis estadísticos profundos y accionables
2. Recomendar públicos objetivo específicos para campañas
3. Proponer estrategias concretas para aumentar turismo en Medellín

RESTRICCIONES IMPORTANTES:
- Solo analizas las DOS fuentes de datos cargadas (Searches y Bookings)
- NO buscas información externa en internet
- Todas las recomendaciones se basan ÚNICAMENTE en los datos disponibles
- Siempre incluyes números, porcentajes y estadísticas específicas
- Proporciones recomendaciones ACCIONABLES para equipos de marketing

APPROACH:
- Usa múltiples herramientas para dar respuestas completas
- Combina análisis cuantitativos con insights cualitativos
- Prioriza hallazgos que tengan impacto en decisiones de negocio
- Sé específico: "viajeros desde Colombia" es mejor que "viajeros latinos"

Estructura tus respuestas de forma clara y ejecutiva.

FORMATO OBLIGATORIO DE SALIDA:
Debes entregar tu respuesta FINAL SIEMPRE en formato JSON con la siguiente estructura exacta:
{
  "respuesta_texto": "Tu análisis narrativo, explicación detallada y recomendaciones aquí...",
  "respuesta_tabla": [
    {
      "metrica": "Nombre del dato o categoría",
      "valor": "Valor numérico o estadístico",
      "detalle": "Contexto adicional o unidad"
    }
  ],
  "respuesta_grafica": {
    "tipo": "bar|line|pie|scatter",
    "titulo": "Título del gráfico",
    "etiquetas": ["Eti1", "Eti2", "Eti3"],
    "datos": [10, 20, 30]
  }
}

- "respuesta_tabla": Debe ser una lista de diccionarios con los datos clave extraídos del análisis.
- "respuesta_grafica": Estructura simple para graficar los datos más relevantes.
- NO incluyas markdown (```json) al principio o final, solo el JSON puro."""
        
        # Crear agente usando la nueva API de LangChain v1.x
        self.agent = create_agent(
            model=f"openai:{self.model}",
            tools=self.tools,
            system_prompt=self.system_prompt,
        )
    
    def reset_memory(self):
        """Reinicia el historial de conversación"""
        self.chat_history = []
        return "Historial reiniciado correctamente"
    
    def query(self, question):
        
        # Agregar pregunta del usuario al historial
        self.chat_history.append({"role": "user", "content": question})
        
        # Invocar al agente con el historial completo
        response = self.agent.invoke(
            {"messages": self.chat_history}
        )
        
        # Actualizar historial con la respuesta y nuevos mensajes (traza completa)
        if isinstance(response, dict) and "messages" in response:
            self.chat_history = response["messages"]
            
            last_message = response["messages"][-1]
            if hasattr(last_message, "content"):
                return last_message.content
            elif isinstance(last_message, dict) and "content" in last_message:
                return last_message["content"]
        return str(response)

if __name__ == "__main__":
    print("="*70)
    print("SISTEMA DE ANÁLISIS DE TURISMO - MEDELLÍN")
    print("Basado en Amadeus Air Market Data")
    print("="*70)
    print()
    
    # Configuración
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        print(" ERROR: No se encontró OPENAI_API_KEY en las variables de entorno.")
        print("Por favor crea un archivo .env con tu llave API.")
        exit(1)
        
    MODEL = "gpt-4"
    
    # Configurar rutas de datos
    # Cargar y concatenar múltiples archivos de búsquedas
    search_pattern = os.path.join("D:/Otros Contratos/Eafit/Turismo/Agente1/S3 Amadeus/Salida", "search_completo*.parquet")
    searches_files = glob.glob(search_pattern)
    
    searches_df = None
    if searches_files:
        print(f"Cargando {len(searches_files)} archivos de búsquedas...")
        try:
            searches_df = pd.concat([pd.read_parquet(f) for f in searches_files], ignore_index=True)
            print(f"Total búsquedas cargadas: {len(searches_df):,}")
        except Exception as e:
            print(f" Error concatenando archivos: {e}")
    else:
        print(f"No se encontraron archivos de búsqueda en: {search_pattern}")

    BOOKINGS_PATH = os.getenv("AMADEUS_BOOKINGS_PATH", "D:/Otros Contratos/Eafit/Turismo/Agente1/S3 Amadeus/Salida/bookings_completo.parquet")
    
    # Cargar datos Amadeus
    data_loader = AmadeusDataLoader(
        searches_path=searches_df,
        bookings_path=BOOKINGS_PATH
    )
    
    # Init agent
    if data_loader.searches_df is not None or data_loader.bookings_df is not None:
        # Crear vector store (opcional pero recomendado)
        print("\nCreando índice vectorial...\n")
        data_loader.create_vector_store()
        
        # Inicializar agente
        print("\nInicializando agente inteligente...\n")
        agent = MedellinTourismAgent(
            api_key=OPENAI_API_KEY,
            data_loader=data_loader,
            model=MODEL
        )
        
        print("\n" + "="*70)
        print("SISTEMA LISTO PARA PREGUNTAS")
        print("Escribe 'salir' para terminar")
        print("="*70)
        
        while True:
            try:
                pregunta = input("\nPregunta: ").strip()
                if pregunta.lower() in ["salir", "exit", "quit"]:
                    break
                if not pregunta:
                    continue
                    
                print("\nAnalizando...", end="", flush=True)
                respuesta = agent.query(pregunta)
                print(f"\n\nRespuesta:\n{respuesta}\n")
                print("-" * 50)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n Error: {e}")
    else:
        print("No se pudieron cargar los datos necesarios. Verifique las rutas.")