# TrafficFlowOptimizer - moduł Backend

# Opis
Moduł jest częścią aplikacji opdpowiedzialną za analize nagrań ruchu ulicznego przekazanych przez użytkownika. Moduł ten nie jest częścią aplikacji działającą samodzielnie, komunikacja między użytkownikiem a tą częścią aplikacji odbywa się za pośrednictwem modułu **Backend**.  

## Wykorzystane zewnętrzne biblioteki
Najważniejsze biblioteki wykorzystywane przez moduł to biblioteka fastApi do komunikacji z innymi częściami apklikacji oraz OpenCV do operacji związanych z edycją nagrania.

# Jak uruchomić moduł
##### Wymagania
* zainstalowany Docker
* zainstalowane narzędzie Docker Compose

##### Instrukcja:
* uruchomić Dockera
* w katalogu projektu: `docker compose up`
