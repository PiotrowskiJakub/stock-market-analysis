require 'net/http'
require "csv"

class StooqApi
  DATA_SOURCE_URL = 'https://stooq.pl/q/d/l/'

  attr_reader :url

  def initialize
    @url = URI.parse(DATA_SOURCE_URL)
  end

  def get_data(stock_symbol:, period: 'd')
    params = {
      s: stock_symbol,
      i: period
    }

    url.query = URI.encode_www_form(params)
    response = Net::HTTP.get_response(url)

    CSV.parse(response.body, headers: true).map do |row|
      row.to_hash
    end
  end
end
