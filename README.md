# Crypto-Strategy-Order-Book-Delta-Volume
Собрал готовый Python-скрипт под Binance Futures с логикой по:

дисбалансу стакана
дельте агрессивных сделок
фильтру по объёму
EMA и ATR для подтверждения тренда и риска
Файл: binance_orderbook_delta_strategy.py

Что внутри:

paper-режим по умолчанию, чтобы можно было безопасно проверить логику
--live для реальной торговли через API Binance
расчёт orderbook imbalance по top-N уровням стакана
расчёт delta через futures_aggregate_trades
округление размера ордера по step size из биржи
простой риск-менеджмент через ATR
Я также проверил, что файл без синтаксических ошибок.

Запуск:

pip install python-binance pandas numpy
python3 binance_orderbook_delta_strategy.py --symbol BTCUSDT --paper
Для live-режима:

export BINANCE_API_KEY="..."
export BINANCE_API_SECRET="..."
python3 binance_orderbook_delta_strategy.py --symbol BTCUSDT --live
