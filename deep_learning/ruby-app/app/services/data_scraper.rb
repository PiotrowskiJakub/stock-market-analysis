require 'net/http'
require 'nokogiri'

class DataScraper
  DATA_SOURCE_URL = 'http://www.money.pl/ajax/gielda/finanse/'
  attr_reader :stock_symbol, :page, :url, :params

  def initialize(stock_symbol: 'PLOPTTC00011')
    @stock_symbol = stock_symbol
    @page = 0
    @url = URI.parse(DATA_SOURCE_URL)
    @params = {
      stock_symbol: stock_symbol,
      page: @page
    }
  end

  def lifetime_data
    objects = []
    loop do
      data = get_data
      objects += data[:data]
      next_page!
      break if !data[:next_page?]
    end
    objects
  end

  private

  def next_page!
    @params[:page] += 1
  end

  def money_api
    @_money_api ||= MoneyApi.new
  end

  def get_data
    money_api.get_data(params)
  end
end
