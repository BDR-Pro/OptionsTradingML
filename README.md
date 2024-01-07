# OptionsTradingML


```diff
+ You Should Use Google Collab Only

```

## Goolge Collab Link

[Google Colab Link](https://colab.research.google.com/drive/1BSyh5BE7Gwzfls4nCV1rrWlTv-9jEp5P?usp=sharing)

## Trading Bot README

OptionsTradingML is a Python trading bot designed for options trading and utilizes machine learning algorithms to make decisions on buying and selling options contracts. The bot supports leveraged trading (up to 2x) and implements strategies such as Moving Average Crossover, Iron Condor, and Butterfly Spread.

## Overview

This Python trading bot is designed for options trading and utilizes machine learning algorithms to make decisions on buying and selling options contracts. The bot supports leveraged trading (up to 2x) and implements strategies such as Moving Average Crossover, Iron Condor, and Butterfly Spread.

**Disclaimer: Trading involves risks, and this bot is provided for educational purposes only. Use it at your own risk, and be sure to thoroughly understand the code and trading strategies before deploying it in a live environment.**

## Features

- High-frequency trading with minute-level data.
- Machine learning models for predicting overall market strategy and choosing between Iron Condor and Butterfly Spread strategies.
- Leveraged trading (up to 2x).
- Implementation of Moving Average Crossover, Iron Condor, and Butterfly Spread strategies.
- Risk management and position sizing based on account value.

## Getting Started

### Prerequisites

- Python 3.6 or later
- [Alpaca API](https://alpaca.markets/) account (for paper trading)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/BDR-Pro/OptionsTradingML.git
   cd OptionsTradingML
   ```

2. **Install dependencies:**

   ```bash
   !pip install -r requirements.txt
   ```

### Configuration

1. Obtain API keys from the [Alpaca dashboard](https://app.alpaca.markets/paper/dashboard/overview).
2. Create a file named `.env` in the root directory of the project.

### Usage

Run the trading bot using:

```bash
python trading_bot.ipynb
```

The bot will start making high-frequency trades based on the implemented strategies and machine learning predictions.

## Strategies

1. **Moving Average Crossover:**
   - Buy when the short-term moving average crosses above the long-term moving average.
   - Sell when the short-term moving average crosses below the long-term moving average.

2. **Iron Condor:**
   - Execute the Iron Condor strategy when machine learning predicts it and specific option-related conditions are met.

3. **Butterfly Spread:**
   - Execute the Butterfly Spread strategy when machine learning predicts it and specific option-related conditions are met.

## Customization

Feel free to customize the bot according to your preferences. You can add more strategies, modify risk management, or enhance machine learning models based on your research.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the functionality or fix any bugs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to [Alpaca](https://alpaca.markets/) for providing a powerful API for algorithmic trading.
- Special thanks to the open-source community for various Python libraries used in this project.

## References

- [Alpaca API Documentation](https://alpaca.markets/docs/api-documentation/)
