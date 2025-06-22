<?xml version="1.0" encoding="UTF-8"?>
<sc version="202001" id="1" name="" frequency="10" steps="0" defaultIntergreenMatrix="0" interstagesUsingMinDurations="true" checkSum="565166112">
  <signaldisplays>
    <display id="1" name="Red" state="RED">
      <patterns>
        <pattern pattern="MINUS" color="#FF0000" isBold="true" />
      </patterns>
    </display>
    <display id="2" name="Red/Amber" state="REDAMBER">
      <patterns>
        <pattern pattern="FRAME" color="#CCCC00" isBold="true" />
        <pattern pattern="SLASH" color="#CC0000" isBold="false" />
        <pattern pattern="MINUS" color="#CC0000" isBold="false" />
      </patterns>
    </display>
    <display id="3" name="Green" state="GREEN">
      <patterns>
        <pattern pattern="FRAME" color="#00CC00" isBold="true" />
        <pattern pattern="SOLID" color="#00CC00" isBold="false" />
      </patterns>
    </display>
    <display id="4" name="Amber" state="AMBER">
      <patterns>
        <pattern pattern="FRAME" color="#CCCC00" isBold="true" />
        <pattern pattern="SLASH" color="#CCCC00" isBold="false" />
      </patterns>
    </display>
  </signaldisplays>
  <signalsequences>
    <signalsequence id="3" name="Red-Red/Amber-Green-Amber">
      <state display="1" isFixedDuration="false" isClosed="true" defaultDuration="1000" />
      <state display="2" isFixedDuration="true" isClosed="true" defaultDuration="1000" />
      <state display="3" isFixedDuration="false" isClosed="false" defaultDuration="5000" />
      <state display="4" isFixedDuration="true" isClosed="true" defaultDuration="3000" />
    </signalsequence>
    <signalsequence id="7" name="Red-Green-Amber">
      <state display="1" isFixedDuration="false" isClosed="true" defaultDuration="1000" />
      <state display="3" isFixedDuration="false" isClosed="false" defaultDuration="5000" />
      <state display="4" isFixedDuration="true" isClosed="true" defaultDuration="3000" />
    </signalsequence>
  </signalsequences>
  <sgs>
    <sg id="1" name="Signal group 1" defaultSignalSequence="7">
      <defaultDurations>
        <defaultDuration display="1" duration="1000" />
        <defaultDuration display="3" duration="5000" />
        <defaultDuration display="4" duration="3000" />
      </defaultDurations>
    </sg>
    <sg id="2" name="Signal group 2" defaultSignalSequence="7">
      <defaultDurations>
        <defaultDuration display="1" duration="1000" />
        <defaultDuration display="3" duration="5000" />
        <defaultDuration display="4" duration="3000" />
      </defaultDurations>
    </sg>
    <sg id="3" name="Signal group 3" defaultSignalSequence="7">
      <defaultDurations>
        <defaultDuration display="1" duration="1000" />
        <defaultDuration display="3" duration="5000" />
        <defaultDuration display="4" duration="3000" />
      </defaultDurations>
    </sg>
    <sg id="4" name="Signal group 4" defaultSignalSequence="7">
      <defaultDurations>
        <defaultDuration display="1" duration="1000" />
        <defaultDuration display="3" duration="5000" />
        <defaultDuration display="4" duration="3000" />
      </defaultDurations>
    </sg>
    <sg id="5" name="Signal group 5" defaultSignalSequence="7">
      <defaultDurations>
        <defaultDuration display="1" duration="1000" />
        <defaultDuration display="3" duration="5000" />
        <defaultDuration display="4" duration="3000" />
      </defaultDurations>
    </sg>
    <sg id="6" name="Signal group 6" defaultSignalSequence="7">
      <defaultDurations>
        <defaultDuration display="1" duration="1000" />
        <defaultDuration display="3" duration="5000" />
        <defaultDuration display="4" duration="3000" />
      </defaultDurations>
    </sg>
    <sg id="7" name="Signal group 7" defaultSignalSequence="7">
      <defaultDurations>
        <defaultDuration display="1" duration="1000" />
        <defaultDuration display="3" duration="5000" />
        <defaultDuration display="4" duration="3000" />
      </defaultDurations>
    </sg>
    <sg id="8" name="Signal group 8" defaultSignalSequence="7">
      <defaultDurations>
        <defaultDuration display="1" duration="1000" />
        <defaultDuration display="3" duration="5000" />
        <defaultDuration display="4" duration="3000" />
      </defaultDurations>
    </sg>
    <sg id="9" name="Signal group 9" defaultSignalSequence="3">
      <defaultDurations>
        <defaultDuration display="1" duration="1000" />
        <defaultDuration display="2" duration="1000" />
        <defaultDuration display="3" duration="5000" />
        <defaultDuration display="4" duration="3000" />
      </defaultDurations>
    </sg>
    <sg id="10" name="Signal group 10" defaultSignalSequence="3">
      <defaultDurations>
        <defaultDuration display="1" duration="1000" />
        <defaultDuration display="2" duration="1000" />
        <defaultDuration display="3" duration="5000" />
        <defaultDuration display="4" duration="3000" />
      </defaultDurations>
    </sg>
    <sg id="11" name="Signal group 11" defaultSignalSequence="3">
      <defaultDurations>
        <defaultDuration display="1" duration="1000" />
        <defaultDuration display="2" duration="1000" />
        <defaultDuration display="3" duration="5000" />
        <defaultDuration display="4" duration="3000" />
      </defaultDurations>
    </sg>
    <sg id="12" name="Signal group 12" defaultSignalSequence="3">
      <defaultDurations>
        <defaultDuration display="1" duration="1000" />
        <defaultDuration display="2" duration="1000" />
        <defaultDuration display="3" duration="5000" />
        <defaultDuration display="4" duration="3000" />
      </defaultDurations>
    </sg>
  </sgs>
  <intergreenmatrices />
  <progs>
    <prog id="1" cycletime="48000" switchpoint="0" offset="0" intergreens="0" fitness="0.000000" vehicleCount="0" name="Signal program 1">
      <sgs>
        <sg sg_id="1" signal_sequence="7">
          <cmds>
            <cmd display="3" begin="0" />
            <cmd display="1" begin="16000" />
          </cmds>
          <fixedstates>
            <fixedstate display="4" duration="0" />
          </fixedstates>
        </sg>
        <sg sg_id="2" signal_sequence="7">
          <cmds>
            <cmd display="3" begin="0" />
            <cmd display="1" begin="16000" />
          </cmds>
          <fixedstates>
            <fixedstate display="4" duration="0" />
          </fixedstates>
        </sg>
        <sg sg_id="3" signal_sequence="7">
          <cmds>
            <cmd display="3" begin="16000" />
            <cmd display="1" begin="24000" />
          </cmds>
          <fixedstates>
            <fixedstate display="4" duration="0" />
          </fixedstates>
        </sg>
        <sg sg_id="4" signal_sequence="7">
          <cmds>
            <cmd display="3" begin="24000" />
            <cmd display="1" begin="40000" />
          </cmds>
          <fixedstates>
            <fixedstate display="4" duration="0" />
          </fixedstates>
        </sg>
        <sg sg_id="5" signal_sequence="7">
          <cmds>
            <cmd display="3" begin="24000" />
            <cmd display="1" begin="40000" />
          </cmds>
          <fixedstates>
            <fixedstate display="4" duration="0" />
          </fixedstates>
        </sg>
        <sg sg_id="6" signal_sequence="7">
          <cmds>
            <cmd display="1" begin="0" />
            <cmd display="3" begin="40000" />
          </cmds>
          <fixedstates>
            <fixedstate display="4" duration="0" />
          </fixedstates>
        </sg>
        <sg sg_id="7" signal_sequence="7">
          <cmds>
            <cmd display="3" begin="0" />
            <cmd display="1" begin="16000" />
          </cmds>
          <fixedstates>
            <fixedstate display="4" duration="0" />
          </fixedstates>
        </sg>
        <sg sg_id="8" signal_sequence="7">
          <cmds>
            <cmd display="3" begin="0" />
            <cmd display="1" begin="16000" />
          </cmds>
          <fixedstates>
            <fixedstate display="4" duration="0" />
          </fixedstates>
        </sg>
        <sg sg_id="9" signal_sequence="7">
          <cmds>
            <cmd display="3" begin="16000" />
            <cmd display="1" begin="24000" />
          </cmds>
          <fixedstates>
            <fixedstate display="4" duration="0" />
          </fixedstates>
        </sg>
        <sg sg_id="10" signal_sequence="7">
          <cmds>
            <cmd display="3" begin="24000" />
            <cmd display="1" begin="40000" />
          </cmds>
          <fixedstates>
            <fixedstate display="4" duration="0" />
          </fixedstates>
        </sg>
        <sg sg_id="11" signal_sequence="7">
          <cmds>
            <cmd display="3" begin="24000" />
            <cmd display="1" begin="40000" />
          </cmds>
          <fixedstates>
            <fixedstate display="4" duration="0" />
          </fixedstates>
        </sg>
        <sg sg_id="12" signal_sequence="7">
          <cmds>
            <cmd display="1" begin="0" />
            <cmd display="3" begin="40000" />
          </cmds>
          <fixedstates>
            <fixedstate display="4" duration="0" />
          </fixedstates>
        </sg>
      </sgs>
    </prog>
  </progs>
  <stages />
  <interstageProgs />
  <stageProgs />
  <dailyProgLists />
</sc>