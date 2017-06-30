require 'net/http'

class MoneyApi
  DATA_SOURCE_URL = 'http://www.money.pl/ajax/gielda/finanse/'
  QUARTERS_PER_PAGE = 4

  attr_reader :url

  def initialize
    @url = URI.parse(DATA_SOURCE_URL)
  end

  def get_data(stock_symbol:, period: 'Q', t: 't', page:)
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

    year_data = [{}, {}, {}, {}]
    call_params = {
      isin: stock_symbol,
      p: period,
      t: t,
      o: page * QUARTERS_PER_PAGE
    }

    response = Net::HTTP.post_form(url, call_params)
    response_body = Nokogiri::HTML(response.body)

    data_rows = response_body.css('tr:not(.ikony)')

    data_rows.each_with_index do |row, row_index|
      attribute = attributes[row_index]

      row.children.css('td, th').each_with_index do |item, item_index|
        year_data[item_index][attribute] = item.text.delete("\n").delete("\t")
      end
    end
    year_data.reject!(&:empty?)

    { data: year_data, next_page?: response_body.css('.rotorR').css('a')[0].attributes['onclick'].value.include?('showFinanceData') }
  end
end
