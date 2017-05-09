require 'net/http'
require 'nokogiri'

class DataScraper
  DATA_SOURCE_URL = 'http://www.money.pl/ajax/gielda/finanse/'
  attr_reader :stock_symbol, :offset, :url, :params

  def initialize(stock_symbol: 'PLOPTTC00011')
    @stock_symbol = stock_symbol
    @offset = 0
    @url = URI.parse(DATA_SOURCE_URL)
    @params = {
      isin: stock_symbol,
      p: 'Q',
      t: 't',
      o: @offset
    }
  end

  def get_all
    objects = []
    loop do
      data = get_data
      objects += data[:data]
      @params[:o] += 4
      break if !data[:next]
    end
    objects
  end

  private

  def get_data
    attributes = %w(
      date
      net_income
      zysk_dzialalnosc
      zysk_brutto
      zysk_netto
      przeplyw_net
      przeplyw_net_dz_op
      przeplyw_net_dz_inw
      przeplyw_net_dz_fin
      aktywa_razem
      zobowiazania
      zobowiazania_dlugoterminowe
      zobowiazania_krotkoterminowe
      kapital_wlasny
      kapital_zakladowy
      liczba_akcji
      book_value
      zysk_per_akcja
      rozwodniona_liczba_akcji
      rozwodniona_book_value
      rozwodniony_zysk_per_akcja
      dywidenda_per_akcja
    )

    objects = [{}, {}, {}, {}]

    response = Net::HTTP.post_form(url, params)
    response_body = Nokogiri::HTML(response.body)

    rows = response_body.css('tr:not(.ikony)')

    rows.each_with_index do |row, row_index|
      attribute = attributes[row_index]

      row.children.css('td, th').each_with_index do |item, item_index|
        objects[item_index][attribute] = item.text.delete("\n").delete("\t")
      end
    end

    { data: objects, next: response_body.css('.rotorR').css('a')[0].attributes['onclick'].value.include?('showFinanceData') }
  end
end
