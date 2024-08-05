const axios = require('axios');
const fs = require('fs');

const API_KEY = ' #My Private API Key';
const BASE_URL = 'https://www.alphavantage.co/query';

//Running the function that searches and extractes the relevant data for a given stock
async function getStockData(symbol, functionType = 'TIME_SERIES_DAILY', outputSize = 'compact') {
    const parameters = {
        function: functionType,
        symbol: symbol,
        apikey: API_KEY,
        outputsize: outputSize
    };

    try {
        const response = await axios.get(BASE_URL, { params: parameters });
        return response.data;
    } catch (error) {
        console.error('Error fetching data:', error);
        return null;
    }
}
//Creating a function that is saves the information scraped to a JSON file
function saveToFile(data, filename = 'stock_information.json') {
    fs.writeFileSync(filename, JSON.stringify(data, null, 4));
}

//Giving an example 
(async () => {
    const stockSymbol = 'META';
    const stockData = await getStockData(stockSymbol);

    if (stockData) {
        console.log('Data fetched successfully');
        saveToFile(stockData);
    }
})();

